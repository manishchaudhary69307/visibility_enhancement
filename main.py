import cv2 as cv,numpy as np, matplotlib.pyplot as plt


def cal_membership_fnc_channel(channel, min_val,max_val):
    # min_val= np.min(channel)
    # max_val= np.max(channel)

    # avoiding division by zero if max and min value is same
    if min_val == max_val:
        return 0

    membership_channel= (channel-min_val)/(max_val-min_val)

    return membership_channel


def identification_operator(channel, tau,pow):
    # calculate_membership= cal_membership_fnc_channel(channel)
    # k1= 2*(calculate_membership**2)
    #
    # if(k1<=tau).any():
    #     print("inside if")
    #     return k1
    # else:
    #     print("inside else")
    #     temp=(1-calculate_membership)**2
    #     return (1-(2*temp))
    identified_array=[]
    print("hjjhhkj")

    for sublist in channel:
        sublist_element=[]
        for element in sublist:
            print("fgdgd",)
            calculate_membership= cal_membership_fnc_channel(element, np.min(channel), np.max(channel))
            k1= 2*(calculate_membership**2)

            if(k1<=tau):

                # print("inside if")
                print(k1,"k1")
                sublist_element.append(k1**pow)
            else:
                temp=(1-calculate_membership)**2
                temp2=1-(2*temp)
                # print("temp2", temp2)
                sublist_element.append(temp2**pow)
            # print("sublist", sublist_element)

        identified_array.append((sublist_element))

    return identified_array





def tun_channel(channel,tau, zeta):
    pow= (tau+zeta)

    # ident_operator_val=(identification_operator(channel, tau))
    ident_operator_val=(identification_operator(channel, tau, pow))

    return ident_operator_val




if __name__=="__main__":
    tau_r=0.5
    tau_g=0.6
    tau_b=0.4
    zeta=0.6
    image_path="images/dusty5.jpg"
    org_image=cv.imread(image_path,)
    image=cv.cvtColor(org_image,cv.COLOR_BGR2RGB)
    R,G,B= cv.split(image)
    tun_r=np.array(tun_channel(R,tau_r,zeta ))
    tun_g=np.array(tun_channel(G,tau_g,zeta))
    tun_b=np.array(tun_channel(B,tau_b,zeta))

    merged_image= cv.merge([tun_b,tun_g,tun_r])
    #
    #
    cv.imshow("merged image", merged_image)

    cv.imshow( "original",org_image )
    cv.waitKey(0)
    cv.destroyAllWindows()

