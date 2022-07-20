import numpy as np
import os,sys

def getVideoId(dataset,vidname):
    if dataset == 'ek':
        return getVideoId_ek(vidname)

    elif dataset == 'bf':
        return getVideoId_bf(vidname)


def getVideoName(dataset,vidid):
    if dataset == 'ek':
        return getVideoName_ek(vidid)
    elif dataset == 'bf':
        return getVideoName_bf(vidid)


def getVideoId_ek(video_name):
    video_name = video_name.split('/')[-1]
    video_id = int(video_name.split('_')[1])
    person_id = int(video_name.split('_')[0][1:])

    video_id = [person_id, video_id]

    
    return video_id

def getVideoName_ek(video_id):
    video_name = "P" + str(video_id[0]).zfill(2) + "_" + str(video_id[1]).zfill(2)
    
    return video_name

def breakfast_name_dicts():
    person_dict = {}
    src_dict = {}
    recipe_dict = {}
    for i in range(0,55):
        if i in list(range(0,10)):
            person_dict['P0' + str(i)] = i 
        else:
            person_dict['P'+str(i)] = i    

    src_dict = {'cam01':1, 'cam02':2, 'stereo01':3, 'webcam01':4, 'webcam02':5}

    recipe_dict = {'cereals':1, 'coffee':2, 'friedegg':3, 'juice':4, 'milk':5, 'pancake':6,'salat':7 , 'sandwich':8 , 'scrambledegg':9 , 'tea':10}

    return person_dict, src_dict, recipe_dict

def getVideoId_bf(video_name):
    video_name = video_name.split('.')[0]
    person_name = video_name.split('_')[0]
    src_name = video_name.split('_')[1]
    recipe_name = video_name.split('_')[3]

    person_dict, src_dict, recipe_dict = breakfast_name_dicts()    
    video_id = [person_dict[person_name], src_dict[src_name], recipe_dict[recipe_name]]
    
    return video_id

def getVideoName_bf(video_id):
    person_dict, src_dict, recipe_dict = breakfast_name_dicts()  
    person_dict = {v:k for k,v in person_dict.items()}
    src_dict = {v:k for k,v in src_dict.items()}
    recipe_dict = {v:k for k,v in recipe_dict.items()}

    video_name = person_dict[video_id[0]] + "_" + src_dict[video_id[1]] + "_" + person_dict[video_id[0]] + "_" + recipe_dict[video_id[2]]
    
    return video_name
    
if __name__ == "__main__":
    vid = getVideoId(sys.argv[1],sys.argv[2])
    reverse = getVideoName(sys.argv[1],vid)
    print(sys.argv[1], vid, reverse)

