import json
import re
sample = {
    "_score": 66,
    "_index": "hashtag_tweets",
    "_source": {
        "tweet": {
            "hashtags": ["materialism", "money", "possessions"], 
            "tweet_id": "0x218443", 
            "text": "When do you have enough ? When are you satisfied ? Is you goal really all about money ?  #materialism #money #possessions <LH>"
            }
        }, 
    "_crawldate": "2015-09-09 09:22:55", 
    "_type": "tweets"
}

def preprocess(content: str):
    content = content.replace('<LH>','')
    content = content.lower()
    # content = re.sub(r'@[a-zA-Z0-9]+', '', content)
    # content = re.sub(r'[^a-zA-Z\s]+', ' ', content)
    content = re.sub(r'[\s]+', ' ', content)
    return content

if __name__ == '__main__':
    tweet_data = {}
    label_tab = {}
    label_cnt = 0
    arr = []
    with open('tweets_DM.json','r') as ft:
        lines = ft.readlines()
        for l in lines:
            tmp = json.loads(l)
            key = tmp['_source']['tweet']['tweet_id']
            content = preprocess(tmp['_source']['tweet']['text'])
            if len(content) == 0:
                print(tmp['_source']['tweet']['text'])
                arr.append(key)
            else:
                tweet_data[key] = {}
                tweet_data[key]['content'] = preprocess(content)
    with open('data_identification.csv','r') as f:
        lines = f.readlines()[1:]
        for l in lines:
            l = l.replace('\n','')
            tmp = l.split(',')
            if tmp[0] in arr:
                print(tmp[1])
            else:
                tweet_data[tmp[0]]['type'] = tmp[1]
    
    with open('emotion.csv','r') as f:
        lines = f.readlines()[1:]
        for l in lines:
            l = l.replace('\n','')
            tmp = l.split(',')
            if tmp[1] not in label_tab:
                label_tab[tmp[1]] = label_cnt
                label_cnt+=1
            if tmp[0] in arr:
                print(tmp[1])
            else:
                tweet_data[tmp[0]]['label'] = label_tab[tmp[1]]
    print(label_tab)
    rst = [[],[],[],[],[],[],[],[]]
    with open('train.csv','w') as f1:
        with open('test.csv','w') as f2:
            f1.write('ID,text,labels\n')
            f2.write('ID,text\n')
            for key in tweet_data:
                if tweet_data[key]['type'] == 'train':
                    rst[int(tweet_data[key]['label'])].append({
                        'id':key,
                        'content':tweet_data[key]['content'],
                        'label':tweet_data[key]['label']
                    })
                    f2.write("{},{},{}\n".format(key,tweet_data[key]['content'],tweet_data[key]['label']))
                else:
                    f2.write("{},{}\n".format(key,tweet_data[key]['content']))
            # maxLen = max(len(rst[0]),len(rst[1]),len(rst[2]),len(rst[3]),len(rst[4]),len(rst[5]),len(rst[6]),len(rst[7]))
            # for i in range(0,maxLen):
            #     if i < len(rst[0]):
            #         f1.write("{},{},{}\n".format(rst[0][i]['id'],rst[0][i]['content'],rst[0][i]['label']))
            #     if i < len(rst[1]):
            #         f1.write("{},{},{}\n".format(rst[1][i]['id'],rst[1][i]['content'],rst[1][i]['label']))
            #     if i < len(rst[2]):
            #         f1.write("{},{},{}\n".format(rst[2][i]['id'],rst[2][i]['content'],rst[2][i]['label']))
            #     if i < len(rst[3]):
            #         f1.write("{},{},{}\n".format(rst[3][i]['id'],rst[3][i]['content'],rst[3][i]['label']))
            #     if i < len(rst[4]):
            #         f1.write("{},{},{}\n".format(rst[4][i]['id'],rst[4][i]['content'],rst[4][i]['label']))
            #     if i < len(rst[5]):
            #         f1.write("{},{},{}\n".format(rst[5][i]['id'],rst[5][i]['content'],rst[5][i]['label']))
            #     if i < len(rst[6]):
            #         f1.write("{},{},{}\n".format(rst[6][i]['id'],rst[6][i]['content'],rst[6][i]['label']))
            #     if i < len(rst[7]):
            #         f1.write("{},{},{}\n".format(rst[7][i]['id'],rst[7][i]['content'],rst[7][i]['label']))