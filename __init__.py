# -*- coding: utf-8 -*-
# coding:=utf-8
import os
import sys
import datetime as dt
import traceback
import socket


reload(sys)
sys.setdefaultencoding('utf-8')


logdepth = 0

def log_depth(newdepth=-1):
    global logdepth
    if newdepth > -1:
        logdepth = newdepth
    return logdepth
    
    
def log_depth_push():
    global logdepth
    logdepth+=1
    

def log_depth_pop():
    global logdepth
    logdepth-=1
    if logdepth < 0:
        logdepth=0


def log_info(msg, depth=-1):
    log("INFO ", msg, depth)


def log_warn(msg, depth=-1):
    log("WARN ", msg, depth)


def log_error(msg, depth=-1):
    log("ERROR", msg, depth)


def log_last_exception(depth=-1):
    if any(sys.exc_info()):
        s = traceback.format_exc()
        log_error(s, depth)
    
    
def log_progress(done, total, showstep=None, depth=-1):
    w = 30 # bar width
    if showstep is None:
        showstep = int(total / w * 2)
        showstep = 1 if showstep == 0 else showstep
    if (done % showstep == 0) or (done >= total-1):
        frac = float(done)/float(total)
        msg = "{}{} | {:>6.1%} ({}/{})".format("#"*int(w*frac), " "*(w-int(w*frac)), frac, done, total)
        log_info(msg, depth)


def log(lvl, msg, depth=-1):

    global logdepth
    if depth == -1:
        depth = logdepth
    else:
        logdepth = depth
        
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "   " * depth
    print (now,"|",lvl,"|",prefix+msg)
    sys.stdout.flush()


