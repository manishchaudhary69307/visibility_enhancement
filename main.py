import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate membership function for a given channel
def cal_membership_fnc_channel(channel):
    min_val = np.min(channel)
    max_val = np.max(channel)

    # Avoid division by zero if max and min values are the same
    if min_val == max_val:
        return np.zeros_like(channel, dtype=np.float32)

    membership_channel = (channel - min_val) / (max_val - min_val)
    return membership_channel


# Function to apply the identification operator
def identification_operator(channel, tau):
    calculate_membership = cal_membership_fnc_channel(channel)
    k1 = 2 * (calculate_membership ** 2)

    # Vectorized operation for the identification operator
    condition = (k1 <= tau)
    result = np.where(condition, k1, 1 - 2 * ((1 - calculate_membership) ** 2))

    return result


# Function to tune the channel based on tau and zeta
def tun_channel(channel, tau, zeta):
    pow_value = tau + zeta
    ident_operator_val = identification_operator(channel, tau) ** pow_value

    # Ensure the values are scaled to 0-255 and converted to uint8
    ident_operator_val_scaled = np.clip(ident_operator_val * 255, 0, 255).astype(np.uint8)
    return ident_operator_val_scaled


# Main function
if __name__ == "__main__":
    tau_r = 0.5
    tau_g = 0.6
    tau_b = 0.4
    zeta = 0.6
    image_path = "images/2.jpg"

    # Read and split the image
    org_image = cv.imread(image_path)
    image = cv.cvtColor(org_image, cv.COLOR_BGR2RGB)
    R, G, B = cv.split(image)

    # Apply the tuning function to each channel
    tun_r = tun_channel(R, tau_r, zeta)
    tun_g = tun_channel(G, tau_g, zeta)
    tun_b = tun_channel(B, tau_b, zeta)

    # Merge the channels back into an image
    merged_image = cv.merge([tun_b, tun_g, tun_r])

    # Display the results
    cv.imshow("Merged Image", merged_image)
    cv.imshow("Original", org_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
