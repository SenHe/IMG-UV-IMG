import copy
import cv2  
import matplotlib
import numpy as np
import argparse
import pickle
import os
import torch
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from PIL import Image

matplotlib.use('TkAgg')

def create_grid(w, h):


    #import ipdb; ipdb.set_trace()

    grid_list = torch.meshgrid([torch.arange(size) for size in [h,w]])

    grid_list = reversed(grid_list)

    grid_list = [grid.float().unsqueeze(0) for dim,grid in enumerate(grid_list)]


    #x = torch.arange(w).view(1, -1).expand(h, -1).float()
    #y = torch.arange(h).view(-1, 1).expand(-1, w).float()

    #grid = torch.stack([y,x], dim=0).float()

    
    return  torch.stack(grid_list, dim=-1).numpy().squeeze(0)

def scale_grid(w, h, grid):

    # de-normalize the grid value to the original value

    grid[:,:,0] = ((grid[:,:,0] + 1.0) / 2.0) * (w-1)

    grid[:,:,1] = ((grid[:,:,1] + 1.0) / 2.0) * (h-1)

    return grid
    

    


def parse_iuv(result):
    i = result['pred_densepose'][0].labels.cpu().numpy().astype(float)
    uv = (result['pred_densepose'][0].uv.cpu().numpy() * 255.0).astype(float)
    iuv = np.stack((uv[1, :, :], uv[0, :, :], i))
    iuv = np.transpose(iuv, (1, 2, 0))
    return iuv


def parse_bbox(result):
    return result["pred_boxes_XYXY"][0].cpu().numpy()


def concat_textures(array):
    texture = []
    for i in range(4):
        tmp = array[6 * i]
        for j in range(6 * i + 1, 6 * i + 6):
            tmp = np.concatenate((tmp, array[j]), axis=1)
        texture = tmp if len(texture) == 0 else np.concatenate((texture, tmp), axis=0)
    return texture


def interpolate_tex(tex):
    # code is adopted from https://github.com/facebookresearch/DensePose/issues/68
    valid_mask = np.array((tex.sum(0) != 0) * 1, dtype='uint8')
    radius_increase = 10
    kernel = np.ones((radius_increase, radius_increase), np.uint8)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    region_to_fill = dilated_mask - valid_mask
    invalid_region = 1 - valid_mask
    actual_part_max = tex.max()
    actual_part_min = tex.min()
    actual_part_uint = np.array((tex - actual_part_min) / (actual_part_max - actual_part_min) * 255, dtype='uint8')
    actual_part_uint = cv2.inpaint(actual_part_uint.transpose((1, 2, 0)), invalid_region, 1,
                               cv2.INPAINT_TELEA).transpose((2, 0, 1))
    actual_part = (actual_part_uint / 255.0) * (actual_part_max - actual_part_min) + actual_part_min
    # only use dilated part
    actual_part = actual_part * dilated_mask

    return actual_part


def get_texture(im, iuv, bbox=None, tex_part_size=200):
    # this part of code creates iuv image which corresponds
    # to the size of original image (iuv from densepose is placed
    # within pose bounding box).
    im = im.transpose(2, 1, 0) / 255
    image_w, image_h = im.shape[1], im.shape[2]
    #bbox[2] = bbox[2] - bbox[0]
    #bbox[3] = bbox[3] - bbox[1]
    #x, y, w, h = [int(v) for v in bbox]
    bg = np.zeros((image_h, image_w, 3))
    ## bounding box based densepose prediction
    #bg[y:y + h, x:x + w, :] = iuv
    ## image size based densepose prediction
    bg = iuv
    iuv = bg
    iuv = iuv.transpose((2, 1, 0))

    #import ipdb; ipdb.set_trace()
    i, u, v = iuv[2], iuv[1], iuv[0]

    # following part of code iterate over parts and creates textures
    # of size `tex_part_size x tex_part_size`
    n_parts = 24
    texture = np.zeros((n_parts, 3, tex_part_size, tex_part_size))
    
    for part_id in range(1, n_parts + 1):
        generated = np.zeros((3, tex_part_size, tex_part_size))

        x, y = u[i == part_id], v[i == part_id]
        # transform uv coodrinates to current UV texture coordinates:
        tex_u_coo = (x * (tex_part_size - 1) / 255).astype(int)
        tex_v_coo = (y * (tex_part_size - 1) / 255).astype(int)
        
        # clipping due to issues encountered in denspose output;
        # for unknown reason, some `uv` coos are out of bound [0, 1]
        tex_u_coo = np.clip(tex_u_coo, 0, tex_part_size - 1)
        tex_v_coo = np.clip(tex_v_coo, 0, tex_part_size - 1)
        
        # write corresponding pixels from original image to UV texture
        # iterate in range(3) due to 3 chanels
        for channel in range(3):
            generated[channel][tex_v_coo, tex_u_coo] = im[channel][i == part_id]
        
        # this part is not crucial, but gives you better results 
        # (texture comes out more smooth)
        if np.sum(generated) > 0:
            generated = interpolate_tex(generated)

        # assign part to final texture carrier
        texture[part_id - 1] = generated[:, ::-1, :]
    
    # concatenate textures and create 2D plane (UV)
    tex_concat = np.zeros((24, tex_part_size, tex_part_size, 3))
    for i in range(texture.shape[0]):
        tex_concat[i] = texture[i].transpose(2, 1, 0)
    tex = concat_textures(tex_concat)

    return tex

def TransferTexture(TextureIm,im,IUV):

    U = IUV[:,:,1]
    V = IUV[:,:,2]
    #
    R_im = np.zeros(U.shape)
    G_im = np.zeros(U.shape)
    B_im = np.zeros(U.shape)
    ###
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        #import ipdb; ipdb.set_trace()
        tex = TextureIm[PartInd-1,:,:,:].squeeze() # get texture for each part.
        #####
        R = tex[:,:,0]
        G = tex[:,:,1]
        B = tex[:,:,2]
        ###############
        #import ipdb; ipdb.set_trace()
        x,y = np.where(IUV[:,:,0]==PartInd)
        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        r_current_points = R[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        g_current_points = G[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        b_current_points = B[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        ##  Get the RGB values from the texture images.
        R_im[IUV[:,:,0]==PartInd] = r_current_points
        G_im[IUV[:,:,0]==PartInd] = g_current_points
        B_im[IUV[:,:,0]==PartInd] = b_current_points
    generated_image = np.concatenate((B_im[:,:,np.newaxis],G_im[:,:,np.newaxis],R_im[:,:,np.newaxis]), axis =2 ).astype(np.uint8)
    #import ipdb; ipdb.set_trace()
    BG_MASK = generated_image==0
    generated_image[BG_MASK] = im[BG_MASK]  ## Set the BG as the old image.
    return generated_image


def main(args):

    densepose_file = os.path.join(args.densepose_path, args.image_name[:-4] + '.png')
    target_densepose_file = os.path.join(args.densepose_path, args.target_name[:-4] + '.png')
    garment_densepose_file = os.path.join(args.garment_densepose_path, args.image_name[:-4] + '.png')
    image_file = os.path.join(args.image_path, args.image_name)
    garment_file = os.path.join(args.garment_path, args.image_name[:-5]+'1.jpg')
    mask_file = os.path.join(args.mask_path,args.image_name[:-5]+'1.jpg')
    
    target_file = os.path.join(args.densepose_path, args.target_name[:-4] + '.png')

    
    iuv = cv2.imread(densepose_file)
    g_iuv = cv2.imread(garment_densepose_file)
    t_iuv = cv2.imread(target_densepose_file)


    mask_g = iuv[:,:,0].astype(float)
    g_MaskBool = np.tile((mask_g==0)[:,:,np.newaxis],[1,1,3])
    g_Mask_vis = cv2.applyColorMap( (mask_g*11).astype(np.uint8) , cv2.COLORMAP_PARULA)[:,:,:]
    g_Mask_vis[g_MaskBool]=0
    cv2.imwrite('./output/garment_densepose_body'+ args.image_name[:-5] + '.png', g_Mask_vis)

    g_umap = cv2.applyColorMap( g_iuv[:,:,1] , cv2.COLORMAP_PARULA)[:,:,:]
    g_umap[g_MaskBool]=0
    
    

    cv2.imwrite('./output/garment_densepose_u'+ args.image_name[:-5] + '.png', g_umap)

    g_vmap = cv2.applyColorMap( g_iuv[:,:,2] , cv2.COLORMAP_PARULA)[:,:,:]
    g_vmap[g_MaskBool]=0
    
    

    cv2.imwrite('./output/garment_densepose_v'+ args.image_name[:-5] + '.png', g_vmap)

    mask = cv2.imread(mask_file)
    mask[mask<128] = 0
    mask[mask>=128] = 1


    
    cv2.imwrite('./output/source_densepose'+'.png', iuv)
    cv2.imwrite('./output/target_densepose'+'.png', t_iuv)
    cv2.imwrite('./output/garment_densepose'+args.image_name[:-5]+'.png', g_iuv)

    
    ##
    image = cv2.imread(image_file)[:,:,::-1]
    garment = cv2.imread(garment_file)[:,:,::-1]
    garment = garment * mask

    #import ipdb; ipdb.set_trace()

    #exa_texture = cv2.imread('texture_from_SURREAL.png')[:,:,::-1]/255.0

    

    ## extract uv texture map
    uv_texture = get_texture(image, iuv[:,:,::-1].astype(float)).transpose(1,0,2)
    garment_uv_texture = get_texture(garment, g_iuv[:,:,::-1].astype(float)).transpose(1,0,2)
    #uv_texture = uv_texture[:,:,::-1]
    cv2.imwrite('./output/source_uvtexture'+'.png', (uv_texture*255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite('./output/garment_uvtexture'+'.png', (garment_uv_texture*255).astype(np.uint8)[:,:,::-1])


    ##texture transfer
    #import ipdb; ipdb.set_trace()



    texture_im = np.zeros([24,200,200,3])
    for i in range(4):

        for j in range(6):

            texture_im[(6*i+j) , :,:,:] = uv_texture[ (200*j):(200*j+200), (200*i):(200*i+200) ,: ]

    texture_garment = np.zeros([24,200,200,3])
    for i in range(4):

        for j in range(6):

            texture_garment[(6*i+j) , :,:,:] = garment_uv_texture[ (200*j):(200*j+200), (200*i):(200*i+200) ,: ]


    #import ipdb; ipdb.set_trace()


    im  = np.zeros(iuv.shape)
    g_im  = np.zeros(g_iuv.shape)
    #import ipdb; ipdb.set_trace()org = label2rgb(org, bg_label=0)
    transfer_image = TransferTexture(texture_im,im,t_iuv)
    transfer_garment = TransferTexture(texture_garment,g_im,t_iuv)
    cv2.imwrite('./output/transfer_image'+'.png', transfer_image)
    cv2.imwrite('./output/transfer_garment'+args.image_name[:-5]+'.png', transfer_garment)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/home/sh0089/sen/fashion/train_img', help='path to the image directory')
    parser.add_argument('--garment_path', type=str, default='/home/sh0089/sen/fashion/train_color', help='path to the image directory')
    parser.add_argument('--mask_path', type=str, default='/home/sh0089/sen/fashion/train_edge', help='path to the image directory')
    parser.add_argument('--grid_path', type=str, default='/home/sh0089/sen/PF-AFN/PF-AFN_test/vton_train_grid', help='path to the image directory')
    parser.add_argument('--densepose_path', type=str, default='/home/sh0089/sen/fashion/train_iuv', help='path to the densepose directory')
    parser.add_argument('--garment_densepose_path', type=str, default='/home/sh0089/sen/fashion/train_garment_iuv', help='path to the densepose directory')
    parser.add_argument('--image_name', type=str, default = '000145_0.jpg', help='image to be visualized')
    parser.add_argument('--target_name', type=str, default = '000079_0.jpg', help='image to be visualized')

    args = parser.parse_args()

    main(args)


