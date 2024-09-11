from sklearn . datasets import fetch_openml
from sklearn . model_selection import train_test_split
from sklearn . metrics import mean_squared_error
from sklearn . tree import DecisionTreeRegressor

# Paragraph C :
# Load dataset
machine_cpu = fetch_openml ( name = 'machine_cpu' )
machine_data = machine_cpu . data
machine_labels = machine_cpu . target
# Split train : test = 8:2
X_train , X_test , y_train , y_test = train_test_split (
machine_data , machine_labels ,
test_size =0.2 ,
random_state =42)

# Paragraph B :
# Define model
tree_reg = DecisionTreeRegressor ()

# Paragraph A :
# Train
tree_reg . fit ( X_train , y_train )

# Paragraph D :
# Preidct and evaluate
y_pred = tree_reg . predict ( X_test )
mean_squared_error ( y_test , y_pred )