import commands
### taken from pubs script, fetch gpu info, not for users to understand
def ls_gpu():
    output = commands.getoutput('nvidia-smi | grep MiB')
    mem_usage = {}
    mem_max   = []
    for l in output.split('\n'):
        words=l.split()
        if len(words) < 4: continue
    
        if words[1].isdigit():
            gpu_id  = int(words[1])
            if not gpu_id in mem_usage:
                mem_usage[gpu_id] = 0
            mem_usage[gpu_id] += int(words[-2].replace('MiB',''))
        else:
            mem_max.append(int(words[-5].replace('MiB','')))

    for i in xrange(len(mem_max)):
        if not i in mem_usage:
            mem_usage[i] = 0

    for i in mem_usage:
        assert i < len(mem_max)

    return (mem_usage,mem_max)

### taken from pubs script, pick an available gpu, not for users to understand
def pick_gpu(mem_min=1,caffe_gpuid=False):
    gpu_info = ls_gpu()
    for gpu in gpu_info[0]:
        #if gpu == 3: continue
        mem_available = gpu_info[1][gpu] - gpu_info[0][gpu]
        if mem_available > mem_min:
            if caffe_gpuid:
                return (len(gpu_info[0])-1) - gpu
            else:
                return gpu
    return -1
