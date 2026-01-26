"""Kubernetes-based bash session backend for GKE production."""
import os
import time
import tempfile
import logging
from typing import Dict, Any, Optional
from kubernetes import client, config
from kubernetes.stream import stream as k8s_stream

from .bash_backend import BashSessionBackend, BashSession

logger = logging.getLogger(__name__)


class KubernetesBashBackend(BashSessionBackend):
    """Kubernetes backend for bash sessions in GKE production."""
    
    def __init__(self, image_name: str, namespace: str = "default", 
                 storage_class: str = "standard-rwo"):
        """
        Initialize Kubernetes backend.
        
        Args:
            image_name: Docker image to use for bash session pods
            namespace: Kubernetes namespace for bash pods
            storage_class: Storage class for PersistentVolumeClaims
        """
        self.image_name = image_name
        self.namespace = namespace
        self.storage_class = storage_class
        
        # Load in-cluster config (works in GKE pods)
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except Exception as e:
            logger.error(f"Failed to load in-cluster config: {e}")
            raise
        
        self.v1 = client.CoreV1Api()
    
    def create_session(self, session_id: str) -> BashSession:
        """Create Kubernetes pod bash session with PersistentVolumeClaim."""
        import uuid
        
        # Generate unique names
        session_id_short = (session_id or str(uuid.uuid4()))[:8]
        pod_name = f"bash-session-{session_id_short}"
        pvc_name = f"bash-workspace-{session_id_short}"
        
        try:
            # Create PVC for shared workspace
            pvc = client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(name=pvc_name),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=["ReadWriteOnce"],
                    storage_class_name=self.storage_class,
                    resources=client.V1ResourceRequirements(
                        requests={"storage": "1Gi"}
                    )
                )
            )
            
            logger.info(f"Creating PVC {pvc_name} for bash session")
            self.v1.create_namespaced_persistent_volume_claim(self.namespace, pvc)
            
            # Wait for PVC to be bound
            for attempt in range(30):
                pvc_status = self.v1.read_namespaced_persistent_volume_claim(pvc_name, self.namespace)
                if pvc_status.status.phase == "Bound":
                    logger.info(f"PVC {pvc_name} is bound")
                    break
                time.sleep(1)
            else:
                raise Exception(f"PVC {pvc_name} failed to bind")
            
            # Create Pod with bash session
            pod = client.V1Pod(
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    labels={"app": "bash-session", "session-id": session_id_short}
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="bash",
                            image=self.image_name,
                            command=["sleep", "infinity"],
                            resources=client.V1ResourceRequirements(
                                limits={"cpu": "1", "memory": "512Mi"},
                                requests={"cpu": "100m", "memory": "128Mi"}
                            ),
                            security_context=client.V1SecurityContext(
                                read_only_root_filesystem=True,
                                allow_privilege_escalation=False
                            ),
                            volume_mounts=[
                                client.V1VolumeMount(
                                    name="workspace",
                                    mount_path="/workspace"
                                ),
                                client.V1VolumeMount(
                                    name="tmp",
                                    mount_path="/tmp"
                                )
                            ]
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="workspace",
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name=pvc_name
                            )
                        ),
                        client.V1Volume(
                            name="tmp",
                            empty_dir=client.V1EmptyDirVolumeSource(size_limit="100Mi")
                        )
                    ],
                    restart_policy="Never"
                )
            )
            
            logger.info(f"Creating pod {pod_name} for bash session")
            self.v1.create_namespaced_pod(self.namespace, pod)
            
            # Wait for pod to be running
            for i in range(60):
                pod_status = self.v1.read_namespaced_pod(pod_name, self.namespace)
                if pod_status.status.phase == "Running":
                    logger.info(f"Pod {pod_name} is running")
                    break
                if pod_status.status.phase == "Failed":
                    raise Exception(f"Pod {pod_name} failed to start")
                time.sleep(1)
            else:
                raise Exception(f"Pod {pod_name} failed to start after 60 seconds")
            
            # Create local temp directory for session
            local_session_temp = os.path.join(tempfile.gettempdir(), f"k8s_session_{session_id_short}")
            os.makedirs(local_session_temp, exist_ok=True)
            
            return BashSession(
                session_id=session_id or "default",
                backend_id=pod_name,
                workspace_path="/workspace",
                session_temp_dir=local_session_temp,
                backend_type="kubernetes"
            )
        
        except Exception as e:
            logger.error(f"Error creating Kubernetes bash session: {e}")
            # Attempt cleanup if creation failed
            try:
                self.v1.delete_namespaced_pod(pod_name, self.namespace, grace_period_seconds=0)
                self.v1.delete_namespaced_persistent_volume_claim(pvc_name, self.namespace, grace_period_seconds=0)
            except:
                pass
            raise
    
    def execute_command(self, session: BashSession, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute command in pod via Kubernetes API."""
        try:
            exec_command = ["/bin/sh", "-c", f"cd /workspace && {command}"]
            
            # Use stream() for kubectl exec equivalent
            resp = k8s_stream(
                self.v1.connect_get_namespaced_pod_exec,
                session.backend_id,
                self.namespace,
                command=exec_command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
                _return_http_data_only=True
            )
            
            stdout_data = []
            stderr_data = []
            
            # Read output with timeout
            start_time = time.time()
            while resp.is_open() and (time.time() - start_time) < timeout:
                resp.update(timeout=1)
                if resp.peek_stdout():
                    stdout_data.append(resp.read_stdout())
                if resp.peek_stderr():
                    stderr_data.append(resp.read_stderr())
            
            resp.close()
            
            return {
                "output": "".join(stdout_data),
                "error": "".join(stderr_data),
                "return_code": 0  # Kubernetes doesn't provide return code directly
            }
        
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "output": "",
                "error": str(e),
                "return_code": 1
            }
    
    def check_session_health(self, session: BashSession) -> bool:
        """Check if pod is running."""
        try:
            pod = self.v1.read_namespaced_pod(session.backend_id, self.namespace)
            is_healthy = pod.status.phase == "Running"
            if is_healthy:
                logger.debug(f"Pod {session.backend_id} is healthy")
            else:
                logger.warning(f"Pod {session.backend_id} is not running (phase: {pod.status.phase})")
            return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def cleanup_session(self, session: BashSession) -> None:
        """Delete pod and PVC."""
        try:
            import uuid
            session_id_short = session.session_id[:8] if session.session_id else str(uuid.uuid4())[:8]
            pvc_name = f"bash-workspace-{session_id_short}"
            
            # Delete pod with immediate grace period
            try:
                self.v1.delete_namespaced_pod(
                    session.backend_id,
                    self.namespace,
                    grace_period_seconds=0
                )
                logger.info(f"Deleted pod {session.backend_id}")
            except Exception as e:
                logger.warning(f"Failed to delete pod: {e}")
            
            # Delete PVC with immediate grace period
            try:
                self.v1.delete_namespaced_persistent_volume_claim(
                    pvc_name,
                    self.namespace,
                    grace_period_seconds=0
                )
                logger.info(f"Deleted PVC {pvc_name}")
            except Exception as e:
                logger.warning(f"Failed to delete PVC: {e}")
        
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
