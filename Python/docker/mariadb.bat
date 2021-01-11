::docker pull mariadb
docker container run -d -p 3306:3306 -e MYSQL_ROOT_PASSWORD=1111 -v D:\Docker\mariadb:/var/lib/mysql --name mariadb mariadb