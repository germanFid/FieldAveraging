from averager import *
from structures import *

# with open("iter_50.out.csv") as file:
#     data1 = StreamData(file.read(), 501, 981, 81)
    
file = open("iter_50.out.csv", "r")
print("OPENED!")

data1 = StreamData(file.read(), 501, 981, 81)

print(data1.dataset)