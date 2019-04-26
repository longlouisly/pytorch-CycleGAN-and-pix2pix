# simple code to count receptive field when using n_layers parameter.
# by default 3 gives 70x70 patch gan, for example.

def f(output_size, ksize, stride):
    return (output_size - 1) * stride + ksize

rf = []
rf.append(f(output_size=1, ksize=4, stride=1))
rf.append(f(output_size=rf[-1], ksize=4, stride=1))

n_layers = 10
for i in range(n_layers):
    rf.append(f(output_size=rf[-1], ksize=4, stride=2))

print(rf)

