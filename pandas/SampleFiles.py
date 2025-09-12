import pandas as pd

# Sample data creation, Dataframe creation
datta = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago'],
    'Country': ['USA', 'USA', 'USA']
}

df = pd.DataFrame(datta)
print(df)


# df = pd.read_csv('datta1.csv') #loading csv file
# df = pd.read_excel('datta1.xlsx') #loading excel file
# df = pd.DataFrame(datta_dict) #loading dictionary