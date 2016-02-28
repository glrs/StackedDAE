import os
import signal
import shlex, subprocess

from tensorflow import tensorboard as tb

from config import FLAGS, home_out

_summary_dir = FLAGS.summary_dir
_tb_pid_file = home_out(".tbpid")

_tb_path = os.path.join(os.path.dirname(tb.__file__), 'tensorboard.py')
_tb_port = "6006"
_tb_host = "0.0.0.0"

def start():
    if not os.path.exists(_tb_path):
        raise EnvironmentError("tensorboard.py not found!")
    
    if os.path.exists(_tb_pid_file):
        tb_pid = int(open(_tb_pid_file, 'r').readline().strip())
        try:
            os.kill(tb_pid, signal.SIGKILL)
        except OSError:
            pass
        
        os.remove(_tb_pid_file)

    devnull = open(os.devnull, 'wb')
    args = shlex.split('nohup ' + FLAGS.python + ' -u ' + _tb_path + ' --host '+ _tb_host
                       + ' --port ' + _tb_port + ' --logdir={0}'.format(_summary_dir))
    
    p = subprocess.Popen(args, stdout=devnull, stderr=devnull)
    
    with open(_tb_pid_file, 'w') as f:
        f.write(str(p.pid))
    
#     if not FLAGS.no_browser:
#         subprocess.Popen(['open', 'http://localhost:{0}'.format(_tb_port)])


if __name__ == '__main__':
    start()
