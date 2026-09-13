from baseline_model import model, history
from preprocessing import x_test, y_test_cat, y_test
import matplotlib.pyplot as plt
import numpy as np

test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training curves
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend(); plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

preds = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

# Look at specific misclassified digits
wrong_idx = np.where(preds != y_test)[0]
print(f"{len(wrong_idx)} misclassified out of {len(y_test)}")