#!/usr/bin/env python
# coding: utf-8

# In[1]:


def fib(number):
    if number == 0:
        return 0
    elif number == 1:
        return 1
    else:
        return fib(number - 1) + fib(number - 2)
    
max_item_input = input("Enter the number of items you want in Fibonacci series\n")
max_item = int(max_item_input)

for count in range(max_item):
    print(fib(count), end="," )


# In[ ]:




