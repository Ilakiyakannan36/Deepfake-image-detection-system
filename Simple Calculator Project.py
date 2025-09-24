# Simple calculator program
#Simple Calculator
def add(a,b):
    return a+b
def sub(a,b):
    return a-b
def mul(a,b):
    return a*b
def div(a,b):
    if b==0:
        return("Division by 0 is not allowed")
    return a/b
print("===Simple Calculator===")
print("Press 'q' to quit.\n")
op = " "
while op!= "q":
    op=input("Enter operation (+,-,*,/ or q to quit):")
    if op == "q" :
        print("\nExisting... Goodbye!\n")
        break
    if op not in ('+','-','*','/'):
        print("Invalid Operator")
        continue
    a= float(input("Enter first number:"))
    b= float(input("Enter second number:"))

    if op =='+':
        print("result = ", add(a,b))
    elif op =='-':
        print("result = " ,sub(a,b))
    elif op =='*':
        print( "result = " ,mul(a,b))
    elif op =='/':
        print("result = " ,div(a,b))
    else:
        print("Invalid")
        
        