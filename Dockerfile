FROM codercom/code-server:latest

WORKDIR /home/coder/project

# uv project setup

USER coder

COPY pyproject.toml /home/coder/project/pyproject.toml
COPY dataset/ /home/coder/project/dataset/
COPY data/ /home/coder/project/data/
COPY tasks-sample.ipynb /home/coder/project/tasks-sample.ipynb
COPY speechscore/ /home/coder/project/speechscore/
COPY emotion_cnn.pt /home/coder/project/emotion_cnn.pt
COPY tasks-sample-demo.ipynb /home/coder/project/tasks-sample-demo.ipynb


# Install dependencies
RUN sudo apt update && sudo apt install -y build-essential
RUN sudo apt install bash-completion

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

# Install vscode extension
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension ms-toolsai.jupyter
# RUN code-server --install-extension amazonwebservices.amazon-q-vscode

# Update APT mirror to ustc.edu.cn
RUN sudo sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources