# 使用轻量级的 JDK 镜像
FROM maven:3.8.5-openjdk-17-slim

# 设置工作目录
WORKDIR /root

# 将测试与目标项目目录复制到镜像根目录
COPY test_projects /root/test_projects
COPY target_projects /root/target_projects

RUN cd /root/test_projects/thumbnailator && mvn clean -DskipTests install
# 先把 pom.xml 拷进去，预下载依赖（这一步是为了加速，不用每次运行都下包）
#COPY pom.xml .
#RUN mvn dependency:go-offline
# 这个容器启动后默认不做任何事，等待指令
CMD ["sleep", "infinity"]