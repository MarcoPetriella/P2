# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:25:49 2018

@author: Marco
"""

class Customer(object):
    """A customer of ABC Bank with a checking account. Customers have the
    following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, name, balance=0.0):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.name = name
        self.balance = balance
        self.variable = 2

    def withdraw(self, amount):
        """Return the balance remaining after withdrawing *amount*
        dollars."""
        if amount > self.balance:
            raise RuntimeError('Amount greater than available balance.')
        self.balance -= amount
        return self.balance
    
    def deposit(self, amount):
        """Return the balance remaining after depositing *amount*
        dollars."""
        self.balance += amount
        return self.balance   
    
    def aumentar(self):
        
        self.deposit(10)
        
    def inicializa_variable(self):
        self.variable1 = 0
        
        self.variable2 = self.variable1 +10
    
    
marco = Customer('marco',30)    
marco.deposit(10)
marco.aumentar()
marco.inicializa_variable()