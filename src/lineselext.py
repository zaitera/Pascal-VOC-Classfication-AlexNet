with open("output.txt", "r") as f:
    lines = f.readlines()
with open("result1222.txt", "w") as f:
    i=0
    epochs = list()
    mAPs = list()
    for line in lines:
        words = line.split()
        if len(words)>0 and words[0] == 'TESTING:':
            i+=5
            f.write("epoch = "+str(i)+"\t"+words[-2]+"= "+words[-1]+'\n')
            epochs.append(i)
            mAPs.append(float(words[-1]))

    print("epochs: ")
    print(epochs)
    print("mAPs: ")
    print(mAPs)
            