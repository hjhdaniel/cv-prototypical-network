import matplotlib.pyplot as plt

def read_txt_to_list(filename):
    with open(filename, 'r') as f:
        ls = [float(line.strip()) for line in f]
    return ls

val_avg_loss = read_txt_to_list('../output/val_avg_loss.txt')
val_avg_acc = read_txt_to_list('../output/val_avg_acc.txt')
feature_val_avg_loss = read_txt_to_list('../feature_output/val_avg_loss.txt')
feature_val_avg_acc = read_txt_to_list('../feature_output/val_avg_acc.txt')


# Plotting graphs
plt.figure()
plt.plot(val_avg_loss, label="Original Loss")
plt.plot(feature_val_avg_loss, label="Feature Loss")
plt.title("Test Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
# plt.show()
plt.savefig('Losses.png')
plt.clf()

plt.figure()
plt.plot(val_avg_acc, label='Original Accuracy')
plt.plot(feature_val_avg_acc, label='Feature Accuracy')
plt.title("Test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig('Test_acc.png')
# plt.show()
plt.clf()