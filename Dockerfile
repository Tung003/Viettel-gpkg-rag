#base image
FROM ubuntu
# update
RUN apt-get update
# install git
RUN apt-get -y install git
# install python
RUN apt-get -y install python3 python3-pip python3-tk libgl1 libglib2.0-0

RUN apt-get clean
#chỉ định thư mục làm việc
WORKDIR /app
#update để cài môi trường
RUN apt update && apt install -y python3.12-venv
# copy file requirements.txt
COPY requirements.txt .

#chạy lệnh cài các thư viện trong requirements.txt
#tạo trong venv thì luồn vào trong venv để cài các thư viện
RUN python3 -m venv venv && \
    venv/bin/pip install -r requirements.txt

#copy hết tất cả các file và folder \
COPY . .
EXPOSE 8000
CMD ["venv/bin/uvicorn", "app.RAG_API:app", "--host", "0.0.0.0", "--port", "8000"]
#run docker run -p 8000:8000 rag_app
