::docker pull mongo
docker container run -d -p 27017:27017 -e MONGO_INITDB_ROOT_USERNAME=root -e MONGO_INITDB_ROOT_PASSWORD=1111 -v D:\Docker\mongodb:/data/db --name mongo mongo