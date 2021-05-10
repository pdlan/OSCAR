FROM nvidia/cuda:10.2-devel-ubuntu18.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
RUN apt-get update && \
	apt-get install -y git cmake golang-go lsb-release wget software-properties-common
RUN wget --quiet https://apt.llvm.org/llvm.sh -O ~/llvm.sh && \
	chmod +x ~/llvm.sh && \
	~/llvm.sh 11 && \
	rm ~/llvm.sh
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
	/bin/bash ~/anaconda.sh -b -p /opt/conda && \
	rm ~/anaconda.sh && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    /opt/conda/bin/conda clean -afy
RUN conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN git clone https://github.com/NVIDIA/apex.git ~/apex && \
	cd ~/apex && \
	bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate && python setup.py install && pip install pyelftools" && \
	rm -rf ~/apex
RUN wget --quiet https://github.com/avast/retdec/releases/download/v4.0/retdec-v4.0-ubuntu-64b.tar.xz -O ~/retdec.tar.xz && \
	tar xf ~/retdec.tar.xz -C /opt/ && \
	ln -s /opt/retdec/bin/retdec-decompiler.py /usr/local/bin/retdec-decompiler && \
	rm ~/retdec.tar.xz
CMD [ "/bin/bash" ]
