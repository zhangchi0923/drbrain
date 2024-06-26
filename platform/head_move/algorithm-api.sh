#!/bin/bash
DIR=/usr/local/project/api/algorithm/algorithm-platform-test/platform/head_move/
LOG=gunicorn-log/
ENTRYPOINT=server.py
CONDA_ENV=py388
SERVICE=algorithm-api
# pid
PID=$DIR$LOG\gunicorn.pid

#使用说明，用来提示输入参数
usage() {
    echo "Usage: sh algorithm-api.sh [start|stop|restart|status]"
    exit 1
}

#检查程序是否在运行
is_exist(){
  pid=`ps -ef|grep gunicorn|grep -v grep|awk 'NR==1{print $2}' `
  #如果不存在返回1，存在返回0     
  if [ -z "${pid}" ]; then
   return 1
  else
    return 0
  fi
}

#启动方法
start(){
  is_exist
  if [ $? -eq "0" ]; then 
    echo ">>> $SERVICE gunicorn service is already running PID=${pid} <<<" 
  else 
    export PATH="~/anaconda3/bin:$PATH"
    source activate
    sleep 3
    echo ">>> Entering conda environment. <<<"
    conda activate $CONDA_ENV
    sleep 1
    echo ">>> Activating $CONDA_ENV. <<<"
    cd $DIR
    /usr/bin/nohup  gunicorn -c gunicorn_conf.py server:app >$DIR\log/console.log 2>&1 &
    # echo $! > $PID
    echo ">>> Start $SERVICE successfully. PID=$!. <<<" 
   fi
  }

  #输出运行状态
status(){
  is_exist
  if [ $? -eq "0" ]; then
    echo ">>> ${SERVICE} is running PID is ${pid} <<<"
  else
    echo ">>> ${SERVICE} is not running <<<"
  fi
}

#停止方法
stop(){
  #is_exist
  pidf=$(cat $PID)
  #echo "$pidf"  
  echo ">>> api PID = $pidf begin kill $pidf <<<"
  kill $pidf
#   rm -rf $PID
  sleep 2
  is_exist
  if [ $? -eq "0" ]; then 
    echo ">>> api 2 PID = $pid begin kill -9 $pid  <<<"
    kill -9  $pid
    sleep 2
    echo ">>> $SERVICE process stopped <<<"  
  else
    echo ">>> ${SERVICE} is not running <<<"
  fi  
}

#重启
restart(){
  stop
  start
}

case "$1" in
    start)
        start
        ;;
    status)
        status
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    *)
        usage
        ;;
esac
exit 0