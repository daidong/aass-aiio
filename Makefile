CUR_DIR:=$(shell pwd)
JAVA:=$(shell which java)
JAVA_VER:=$(shell java -version 2>&1)
MAVEN:=$(shell which mvn)
MAVEN_VER:=$(shell mvn -version)
JAVA_SRC_HOME=$(CUR_DIR)

.PHONY: print_info
print_info:
		# Here is the info corresponding to the essentials for building your system.
		# CUR_DIR	=	$(CUR_DIR);
		# JAVA		=	$(JAVA);
		# JAVA_VER	=	$(JAVA_VER);
		# MAVEN		=	$(MAVEN);
		# MAVEN_VER	=	$(MAVEN_VER);

.PHONY: clean
clean:
		cd $(JAVA_SRC_HOME) && mvn clean;
		-rm -rf release

.PHONY: install
install: clean
		cd $(JAVA_SRC_HOME) && mvn install;
		mkdir -p release; mv target/*.tar.gz release/; cd release && tar zxf aiio-0.1-make-assembly.tar.gz;

.PHONY: all
all: install
		# Start building
