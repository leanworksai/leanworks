from setuptools import setup, find_packages

setup(
    name="leanworks",
    version="0.2.5",
    packages=find_packages(),
    install_requires=[
        "pinecone-client==5.0.1",
        "google-cloud-storage==2.19.0",
        "google-cloud-secret-manager==2.22.0",
        "google-auth==2.37.0",
        "google-auth-oauthlib==1.2.1",
        "google-auth-httplib2==0.2.0",
        "anthropic==0.49.0",
        "google-genai==1.8.0",
        "openai==1.60.0",
        "google-api-python-client==2.159.0",
        "google-cloud-storage==2.19.0",
        "numpy==1.26.0",
        "tiktoken==0.9.0",
        "google-cloud-bigquery==3.32.0"
    ],
    author="Yanfu Zhu",
    author_email="yanfu@leanworks.ai",
    description="Internal LeanWorks package",
    python_requires=">=3.10",
    url="https://github.com/LeanWorks-ai/leanworks",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 