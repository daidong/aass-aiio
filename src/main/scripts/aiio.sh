#!/bin/bash
#
#
PRG="$0"

while [ -h "$PRG" ] ; do
  ls=`ls -ld "$PRG"`
  link=`expr "$ls" : '.*-> \(.*\)$'`
  if expr "$link" : '/.*' > /dev/null; then
    PRG="$link"
  else
    PRG=`dirname "$PRG"`/"$link"
  fi
done

PRGDIR=`dirname "$PRG"`
[ -z "$PROCESSOR_HOME" ] && PROCESSOR_HOME=`cd "$PRGDIR/.." ; pwd`

APP_NAME=AIIO

# path
BIN_PATH=$PROCESSOR_HOME/bin
LOG_PATH=$PROCESSOR_HOME/logs
LIB_PATH=$PROCESSOR_HOME/lib
#
mkdir -p $LOG_PATH
touch $LOG_PATH/stdout.log

#
CLASS_NAME_AIIO=edu.ttu.aass.aiio.AIIOMain
CLASS_NAME_WORD2VEC=edu.ttu.aass.aiio.vectorize.YahooHDFSFile2Vec
CLASS_NAME_LSTM=edu.ttu.aass.aiio.lstm.YahooHDFSLSTM
CLASS_NAME_TEXT_GUI=edu.ttu.aass.aiio.vectorize.TextWord2VecGUI

CLASS_PATH=$PROCESSOR_HOME/conf
#
for f in $LIB_PATH/*.jar
do
    CLASS_PATH=$CLASS_PATH:$f;
done

DEBUG_ARGS="";
#
PROGRAM_ARGS="-Xms96g -Xmx96g -Dapp.name=${SERVER_NAME} -Dapp.base=${PROCESSOR_HOME} -XX:+UseConcMarkSweepGC -server -XX:SurvivorRatio=5 -XX:CMSInitiatingOccupancyFraction=80 -XX:+PrintTenuringDistribution  -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+PrintGCApplicationStoppedTime -XX:+PrintGCApplicationConcurrentTime ${DEBUG_ARGS} -Xloggc:./gc.log"

if [ "AIIO" = $1 ]; then
	java $PROGRAM_ARGS -classpath $CLASS_PATH $CLASS_NAME_AIIO ${@:2}
fi

if [ "SORT" = $1 ]; then
	java $PROGRAM_ARGS -classpath $CLASS_PATH $CLASS_NAME_SORT ${@:2}
fi

if [ "WORD2VEC" = $1 ]; then
	java $PROGRAM_ARGS -classpath $CLASS_PATH $CLASS_NAME_WORD2VEC ${@:2}
fi

if [ "LSTM" = $1 ]; then
	java $PROGRAM_ARGS -classpath $CLASS_PATH $CLASS_NAME_LSTM ${@:2}
fi

if [ "W2VGUI" = $1 ]; then
    java $PROGRAM_ARGS -classpath $CLASS_PATH $CLASS_NAME_TEXT_GUI ${@:2}
fi