import xml.etree.ElementTree as ET
import os
import cv2

'''
Img_dir = 'VOC_2012_DS\JPEGImages'
ann_dir = 'VOC_2012_DS\Annotations'
'''    
def Dataset(Img_dir = 'VOC_2012_DS\JPEGImages', ann_dir = 'VOC_2012_DS\Annotations'):

    ''' It gives output numpy array of shape [n_img, img_ht, img_wt, img_depth]
    and another numpy array - ground truth boxes - of shape [n_img , 4] '''
    
    image_list = []
    bbox_list  = []
    
    for xml_file in os.listdir(ann_dir):
        try :
            bboxes = []
            Image = None
            xml_file_dir = os.path.join(ann_dir, xml_file)
            #Reading xml files one-by-one
            tree = ET.parse(xml_file_dir)
            root = tree.getroot()
            for sroot in root:
                # Reading image file
                if sroot.tag == 'filename':
                    img_name = sroot.text
                    img_dir  = os.path.join(Img_dir, img_name)
                    Image    = cv2.imread(img_dir, 1)
                    print('Reading image file')
                # Getting scaled image with 600 as 
                # length of smaller side                                            
                if sroot.tag == 'size':
                    width = float(sroot[0].text)
                    height= float(sroot[1].text)
                    
                    if width >= height:
                        scale = 600.0/height
                        height= 600
                        width = scale*width
                    else:
                        scale = 600.0/width
                        width = 600
                        height= scale*width
                        
                    Image = cv2.resize(Image,(int(width),int(height)))
                    print('Getting scaled image')
                # Getting bounding-boxes 
                # co-ordinates
                if sroot.tag == 'object':
                    for ssroot in sroot:
                        if ssroot.tag == 'bndbox':
                            bbox = [int(float(ssroot[0].text)*scale),int(float(ssroot[1].text)*scale),
                                    int(float(ssroot[2].text)*scale),int(float(ssroot[3].text)*scale)]
                            bboxes += [bbox]
            image_list += [Image]
            bbox_list  += [bboxes]
        except:
          print('Loading Error :-(')  
          pass
        
    return image_list, bbox_list

if __name__=='__main__':
    
    image_list, bbox_list = Dataset()
    for image in image_list:
        cv2.imshow('IMAGE_SHOW', image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

