from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    N = len(digits_dataset_X)
    print(N)

    # Prints the 64 attributes of a random digit, its class,
    # and then shows the digit on the screen
    digit_to_show = np.random.choice(range(N), 1)[0]
    print("Attributes:", digits_dataset_X[digit_to_show])
    print("Class:", digits_dataset_y[digit_to_show])

    plt.imshow(np.reshape(digits_dataset_X[digit_to_show], (8, 8)), cmap='gray')
    plt.show()
