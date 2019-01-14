import matplotlib.pyplot as plt

def plot_train_process(args,item):
    train_acc_arr,val_acc_arr=item
    plt.figure()
    x=range(len(train_acc_arr))
    
    plt.plot(x,train_acc_arr,'o-',color='green',label='train')
    plt.plot(x,val_acc_arr,'o-',color='red',label='val')
    
    plt.legend()
    plt.savefig(args.train_img,bbox_inches='tight', pad_inches=0)
    plt.close()
    
def plot_different_figs(plot_dict):
    cols=len(plot_dict.keys())
    fig=plt.figure(figsize=(10,20))
    
    cur_row=0
    if plot_dict['raw_imgs']:
        for index,raw in enumerate(plot_dict['raw_imgs']):
            fig.add_subplot(len(plot_dict['raw_imgs']),cols,index*cols+cur_row+1)
            plt.imshow(plot_dict['raw_imgs'][index])
            plt.axis('off')
    
    cur_row+=1
    if 'cams' in plot_dict.keys():
        for index,cam in enumerate(plot_dict['cams']):
            fig.add_subplot(len(plot_dict['cams']),cols,index*cols+cur_row+1)
            plt.imshow(plot_dict['cams'][index].data.numpy())
            plt.axis('off')
            
    plt.savefig('save/img/raw_cam.png',bbox_inches='tight', pad_inches=0)
    plt.close()