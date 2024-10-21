FROM mcr.microsoft.com/devcontainers/base:jammy

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
   && apt-get -y install \
      gcc g++ \
      mpich openmpi-bin libopenmpi-dev libomp-dev \
      cppcheck valgrind \
      python3-pip \
      clangd \
   && apt-get autoremove -y && apt-get clean -y

COPY ./reinstall-cmake.sh /tmp/
RUN chmod +x /tmp/reinstall-cmake.sh \
   && /tmp/reinstall-cmake.sh 3.25.0 \
   && rm -f /tmp/reinstall-cmake.sh

RUN pip install --no-cache-dir xlsxwriter==3.2.0