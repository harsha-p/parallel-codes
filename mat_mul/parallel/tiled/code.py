import subprocess as sp


SIZE=1024
LOOP_COUNT=5
TILING_LEVEL=1
TILE1 = 64
TILE2 = 64
TILE3 = 64

cmd_256=["./release/single_node_256",SIZE,LOOP_COUNT,TILING_LEVEL,TILE1,TILE2,TILE3]
cmd_512=["./release/single_node_512",SIZE,LOOP_COUNT,TILING_LEVEL,TILE1,TILE2,TILE3]

run_times = []

def func( local_loop, local_level, local_t1, local_t2, local_t3):
    cmd_512=["./release/single_node_512",SIZE,local_loop,local_level,pow(2,local_t1),pow(2,local_t2),pow(2,local_t3)]
    cmd_512=[str(x) for x in cmd_512]
    process_512 = sp.run(args=cmd_512,stderr=sp.PIPE,stdout=sp.PIPE,stdin=None)
    return_value_512 = process_512.stderr.decode('utf-8')
    values = return_value_512.split()
    # print(values,local_loop,local_level,pow(2,local_t1),pow(2,local_t2),pow(2,local_t3))
    if(len(values) == 2):
        return func(int(values[1]),local_level,local_t1,local_t2,local_t3)
    else:
        return [float(values[0]),local_loop,local_level,pow(2,local_t1),pow(2,local_t2),pow(2,local_t3)]


for local_level in range(0,4):
    for local_t1 in range(5,11):
        for local_t2 in range(5,local_t1+1):
            for local_t3 in range(5,local_t2+1):
                local_loop = LOOP_COUNT
                lol =  func(local_loop,local_level,local_t1,local_t2,local_t3);
                run_times.append(lol)
                lol = [str(x) for x in lol]
                lol = " ".join(lol)
                lol+='\n'
                file = open("py_output","a")
                file.write(lol)
                file.close()

for i in run_times:
    print(i)
