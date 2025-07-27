FROM codercom/code-server:latest

WORKDIR /home/coder/project

# uv project setup

COPY pyproject.toml /home/coder/project/pyproject.toml

# Install dependencies
RUN sudo apt update && sudo apt install -y build-essential

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Config uv mirror
# Write the following line to ~/.config/uv/uv.toml
# [[index]]
# url = "https://mirrors.ustc.edu.cn/pypi/simple"
# default = true
RUN mkdir -p /home/coder/.config/uv && \
    echo '[[index]]\nurl = "https://mirrors.ustc.edu.cn/pypi/simple"\ndefault = true' > /home/coder/.config/uv/uv.toml

RUN /home/coder/.local/bin/uv sync --upgrade