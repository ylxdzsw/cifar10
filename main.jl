using OhMyJulia
using PyCall
using Fire

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__)

function read_cifar_bin(file, batchsize=20)
    batches, buf = [], Bytes(3073)
    open(file) do f
        while !eof(f)
            X, y = Array{u8}(batchsize, 3, 32, 32), Bytes(batchsize)
            for i in 1:batchsize
                read!(f, buf)
                y[i] = buf[1]
                X[i, 1, :, :] = buf[2:1025]
                X[i, 2, :, :] = buf[1026:2049]
                X[i, 3, :, :] = buf[2050:3073]
            end
            push!(batches, (X, y))
        end
    end
    batches
end

function read_data(dir="D:/cifar-10/raw/")
    data = vcat([read_cifar_bin(dir*"data_batch_$i.bin") for i in 1:5]...)
    X, y = read_cifar_bin(dir*"test_batch.bin", 10000)[]
    data, X, y
end

function lr(i)
    i < 10 ? [.001, .01, .02, .05, .15, .5, .5, .5, .5][i] : 2. ^ -(i ÷ 10)
end

@main function main(modelname; epoch::Int=50, baselr::f64=.01)
    data, X, y = read_data()
    model = pywrap(pyimport(modelname)[:Model]("gpu"))

    for i in 1:epoch
        tic()
        loss = model.train(data, baselr * lr(i))
        acc = count(model.predict(X) .== y) / length(y)
        println("epoch $i, loss: $loss, acc:$acc, time:$(toq())")
        i % 10 == 0 && model.save("D:/cifar-10/result/$modelname.model")
    end
end