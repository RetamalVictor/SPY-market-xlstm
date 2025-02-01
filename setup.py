from setuptools import setup, find_packages

setup(
    name="hft_model",
    version="0.1.0",
    description="A PyTorch Lightning framework for high-frequency trading model training and inference",
    author="Victor Retamal",
    author_email="retamal1.victor@gmail.com",
    packages=find_packages(),
    install_requires=[
        "fastparquet==2024.11.0",
        "numpy==2.1.2",
        "pandas==2.2.3",
        "PyYAML==6.0.2",
        "pytorch-lightning==2.5.0.post0",
        "tensorboard==2.18.0",
        "torch==2.6.0+cu126",
        "tqdm==4.67.1"
    ],
    entry_points={
        "console_scripts": [
            "train=hft_model.training:main",      # Run training with: train --config train_config.yaml
            "infer=hft_model.inference:main",     # Run inference with: infer --checkpoint <ckpt> --data_file <file> [--evaluate]
        ],
    },
    include_package_data=True,  # Includes files specified in MANIFEST.in (if needed)
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
