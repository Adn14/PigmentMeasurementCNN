# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 10:14:19 2019

@author: Adn
"""

from __future__ import print_function
import os
from flask import Flask, request, render_template, send_from_directory, Response, redirect, url_for
import cv2


apDis = {"Apple scab":0, "Black rot":1, "Cedar apple rust":2, "Healthy":3}
apSym = {'Lesions on the leaves and fruit are generally blistered and "scabby" in appearance, with a distinct margin.The earliest noticeable symptom on fruit is water-soaked areas which develop into velvety, green to olive-brown lesions. Infections of young fruit will cause fruit distortion.': 0,
     'The black rot pathogen infects limbs, trunks, leaves and fruit resulting in reduced productivity and quality of yield. Leaf infections result in a disease called frog-eye leaf spot. On leaves, the disease first appears as a tiny purple fleck which eventually enlarges into a circular lesion about 4-5 mm in diameter. The disease often first shows up one to three weeks after petal fall. As the lesion enlarges, the margin remains purple while the centre turns tan or brown with a light centre giving the lesion a "frog-eye" appearance.': 1,
     'Symptoms on the juniper hosts often include swollen growths or woody galls on branches or shoots. Once mature, bright orange, gelatinous, spore-producing growths emerge from the galls. This may be apparent for a period of just a few to several weeks each spring depending on the weather. Mature galls are generally easier to find after a moist period as moisture causes the gelatinous material to swell and appear orange in color. On leaves of rosaceous hosts, lesions may be apparent as yellow or orange spots on the upper surfaces of the leaves.': 2,
     'Healthy': 3}
apTreat = {'new infections can be reduced by removing leaf litter and trimmings containing infected tissue from the orchard and incinerating them. This will reduce the amount of new ascospores released in the spring. Additionally, scab lesions on woody tissue can be excised from the tree if possible and similarly destroyed.':0,
    'Treating black rot on apple trees starts with sanitation. Because fungal spores overwinter on fallen leaves, mummified fruits, dead bark and cankers, it’s important to keep all the fallen debris and dead fruit cleaned up and away from the tree. During the winter, check for red cankers and remove them by cutting them out or pruning away the affected limbs at least six inches beyond the wound. Destroy all infected tissue immediately and keep a watchful eye out for new signs of infection.':1,
    'Remove galls from infected junipers. In some cases, juniper plants should be removed entirely. Apply preventative, disease-fighting fungicides labeled for use on apples weekly, starting with bud break, to protect trees from spores being released by the juniper host.':2,
    '':3}
apCause = {'Apple scab is a disease of Malus trees, such as apple trees, caused by the ascomycete fungus Venturia inaequalis. The disease manifests as dull black or grey-brown lesions on the surface of tree leaves, buds or fruits. Lesions may also appear less frequently on the woody tissues of the tree. Fruits and the undersides of leaves are especially susceptible. The disease rarely kills its host, but can significantly reduce fruit yields and fruit quality. Affected fruits are less marketable due to the presence of the black fungal lesions.':0,
    'Black rot is an important disease of apple caused by the fungus Botryosphaeria obtusa. Black rot fungus infects a wide variety of hardwood trees, including apple and pear. Infected trees are often a source of infection for nearby younger bearing blocks.':1,
    'Cedar apple rust is caused by a fungal pathogen called Gymnosporangium juniperi-virginianae. This fungus attacks crabapples and apples (Malus sp.) and eastern red cedar (Juniper).':2,
    '':3}

corDis = {"Gray leaf spot":0, "Common rust":1, "Healthy":2, "Northern corn leaf blight":3}
corSym = {'Lesions begin as flecks on leaves that develop into small tan spots. Spots turn into elongated brick-red to cinnamon-brown pustules with jagged appearance. Early lesions of gray leaf spot are small, necrotic spots and may have a chlorotic halo (more visible when leaf is backlit). Gray leaf spot can become severe in favorable environmental conditions: 70-90° F, high relative humidity, and prolonged leaf wetness from rain, dew, or irrigation.':0,
         'Disease lesions reduce functional leaf area and photosynthesis. Less sugars are produced, so plant uses stalk carbohydrates to help fill kernels. Stalks are weakened and stalk rot potential increases.':1,
         '':2,
         'Early northern corn leaf blight lesions are gray-green and elliptical, beginning 1 to 2 weeks after infection. Northern corn leaf blight lesions become pale gray to tan as they enlarge to 1 to 6 inches or longer. Distinct cigar-shaped lesions unrestricted by leaf veins make northern corn leaf blight (NCLB) one of the easiest diseases to identify. Under moist conditions, lesions produce dark gray spores, usually on the lower leaf surface, giving the lesions a "dirty" appearance. Heavy blighting and lesion coalescence give leaves a gray/burned appearance.':3}
corTreat = {'Fungicide application is the only management strategy available after planting to lessen the impact of foliar disease outbreaks in corn.':0,
         'Pick off and destroy infected leaves and frequently rake under plants to remove all fallen debris. Water in the early morning hours — avoiding overhead sprinklers — to give plants time to dry out during the day. Use a slow-release, organic fertilizer on crops and avoid excess nitrogen. Apply copper sprays or sulfur powders to prevent infection of susceptible plants. Effectively treat fungal diseases with SERENADE Garden. This broad spectrum bio-fungicide uses a patented strain of Bacillus subtilis and is approved for organic gardening. ':1,
         '':2,
         'Crop rotation to reduce previous corn residues and disease inoculum. Tillage to help break down crop debris and reduce inoculum load. Fungicide application to reduce yield loss and improve harvestability. Consider hybrid susceptibility, previous crop, tillage, field history, application cost, corn price':3}
corCause = {'Fungal disease caused by Cercospora zeae-maydis pathogen.':0,
         'Fungal disease caused by Puccinia sorghi pathogen. Favored by moist, cool conditions (temps in the 60s and 70s). Typically progresses as corn matures in late summer if conditions are persistently wet and cool.':1,
         '':2,
         'Caused by Exserohilum turcicum (previously classified as Helminthosporium turcicum), a fungus found in humid climates wherever corn is grown. Favored by heavy dews, frequent showers, high humidity and moderate temperatures. Spores are spread by rain splash and air currents to the leaves of new crop plants in spring and early summer. Spores may be carried long distances by the wind. Infection occurs when free water is present on the leaf surface for 6 to 18 hours and temperatures are 65 to 80 F.':3}

potDis = {"Early blight":0, "Healthy":1, "Late Blight":2}
potSym = {"Symptoms first appear on the lower, older leaves as small brown spots with concentric rings that form a “bull’s eye” pattern. As the disease matures, it spreads outward on the leaf surface causing it to turn yellow, wither and die. Eventually the stem, fruit and upper portion of the plant will become infected. Crops can be severely damaged.":0,
         "":1,
         "Late blight first appears on the lower, older leaves as water-soaked, gray-green spots. As the disease matures, these spots darken and a white fungal growth forms on the undersides. Eventually the entire plant will become infected. Crops can be severely damaged.":2}
potTreat = {"Prune or stake plants to improve air circulation and reduce fungal problems. Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut. Keep the soil under plants clean and free of garden debris. Add a layer of organic compost to prevent the spores from splashing back up onto vegetation. Drip irrigation and soaker hoses can be used to help keep the foliage dry. For best control, apply copper-based fungicides early, two weeks before disease normally appears or when weather forecasts predict a long period of wet weather. Alternatively, begin treatment when disease first appears, and repeat every 7-10 days for as long as needed. Remove and destroy all garden debris after harvest and practice crop rotation the following year. Burn or bag infected plant parts. Do not compost.":0,
         "":1,
         "Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation. Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation. Destroy all tomato and potato debris after harvest. Apply a copper based fungicide (2 oz/ gallon of water) every 7 days or less, following heavy rain or when the amount of disease is increasing rapidly.":2}
potCause = {"early blight is caused by the fungus Alternaria solani.":0,
         "":1,
         "Late blight is caused by the fungus Phytophthora infestans. ":2}

tomDis = {"Bacterial spot":0,
          "Early blight":1,
          "Healthy":2,
          "Late blight":3,
          "Leaf mold":4,
          "Septoria leaf spot":5,
          "Spider mites":6,
          "Target spot":7,
          "Mosaic Virus":8,
          "Yellow leaf curl virus":9}

tomSym = {"Symptoms of bacterial spot first appear as small, greasy, and irregular marks under the tomato plant’s leaves. The spots start as a dark green, then gradually become purple and gray with black centers, possibly within a white or yellow outer circle. Bacterial spot may result in thin and cracked leaf tissue, exposing the fruit and resulting in sun-scalded output. There is also a risk that the plant might defoliate. Fruit lesions are small dark brown bumps that sink into the fruit as it grows, making it appear scabbed.":0,
          "Symptoms first appear on the lower, older leaves as small brown spots with concentric rings that form a “bull’s eye” pattern. As the disease matures, it spreads outward on the leaf surface causing it to turn yellow, wither and die. Eventually the stem, fruit and upper portion of the plant will become infected. Crops can be severely damaged.":1,
          "":2,
          "Late blight first appears on the lower, older leaves as water-soaked, gray-green spots. As the disease matures, these spots darken and a white fungal growth forms on the undersides. Eventually the entire plant will become infected. Crops can be severely damaged.":3,
          "The tops will begin to develop small, white, gray, yellow, or pale green patches. The underside of the leave will have a fuzzy feel that is purple in color. Sometimes the fuzz will be olive green. The “fuzz” is actually spores of the fungus. The infected tissue becomes yellowish-brown and the leaf begins to wither, eventually falling off of the plant. Untimately the whole plant will wither and die.In the rare case that the blossom or fruit is infected, the mold will appear as a black sore-like lesion. The black can grow to cover over half of the surface area of the plant or blossom. Tomato leaf mold affects ripe tomatoes as well as green tomatoes.":4,
          "Septoria leaf spot usually appears on the lower leaves after the first fruit sets. Spots are circular, about 1/16 to 1/4 inch in diameter with dark brown margins and tan to gray centers with small black fruiting structures. Characteristically, there are many spots per leaf. This disease spreads upwards from oldest to youngest growth. If leaf lesions are numerous, the leaves turn slightly yellow, then brown, and then wither. Fruit infection is rare.":5,
          "Spider mites are among the most ubiquitous of pests, attacking a wide variety of field, garden, greenhouse, nursery, and ornamental plants, as well as several weed species. Infestations of two-spotted spider mites result in the bleaching and stippling of leaves. Severe infestations may cause entire leaves to become bronzed, curled, and completely enveloped in sheets of webbing. Damage to the foliage may result in leaf drop and reduction in the overall vitality of the plant. When a leaf or branch is tapped over a white sheet of paper, the mites appear as small specks that resemble dust or pepper and may be seen to move.":6,
          "On tomato leaves and stems, foliar symptoms of target spot consist of brown-black lesions with subtle concentric rings giving them a target-like appearance. These can sometimes be confused with early blight. On tomato fruit, lesions are more distinct. Small, brown, slightly sunken flecks are seen initially and may resemble abiotic injury such as sandblasting. As fruits mature, the lesions become larger and coalesce, resulting in large pitted areas. Advanced symptoms include large deeply sunken lesions, often with visible dark-gray to black fungal growth in the center. A zone of wrinkled looking tissue may surround the margins of lesions on mature fruit.":7,
          "Light and dark green mottled areas will appear on the leaves of tomato plants infected with this virus, Other symptoms include stunted growth, fruit deformities, and a reduction in the amount of fruit produced.":8,
          "Infected tomato plants initially show stunted and erect or upright plant growth; plants infected at an early stage of growth will show severe stunting. Leaves of infected plants are small and curl upward; and show strong crumpling and interveinal and marginal yellowing. The internodes of infected plants become shortened and, together with the stunted growth, plants often take on a bushy appearance, which is sometimes referred to as 'bonsai' or broccoli'-like growth. Flowers formed on infected plants commonly do not develop and fall off (abscise). Fruit production is dramatically reduced, particularly when plants are infected at an early age, and it is not uncommon for losses of 100% to be experienced in fields with heavily infected plants.":9}

tomTreat = {"To avoid bacterial spot, cultivators should buy certified disease-free tomato seeds and use sterilized soil or a mix that is commercially rendered. If it is not possible to acquire disease-free tomato seeds, your seeds should be submerged for one minute in 1.3'%' sodium hypochlorite, which helps eliminate bacteria on their surface. Another option exists in submerging the seeds in 122-degree Fahrenheit water for 25 minutes. This will affect surface and inner seed bacteria, but might adversely affect the plant’s germination.":0,
          "Prune or stake plants to improve air circulation and reduce fungal problems. Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut. Keep the soil under plants clean and free of garden debris. Add a layer of organic compost to prevent the spores from splashing back up onto vegetation. Drip irrigation and soaker hoses can be used to help keep the foliage dry. For best control, apply copper-based fungicides early, two weeks before disease normally appears or when weather forecasts predict a long period of wet weather. Alternatively, begin treatment when disease first appears, and repeat every 7-10 days for as long as needed. Remove and destroy all garden debris after harvest and practice crop rotation the following year. Burn or bag infected plant parts. Do not compost.":1,
          "":2,
          "Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation. Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation. Destroy all tomato and potato debris after harvest. Apply a copper based fungicide (2 oz/ gallon of water) every 7 days or less, following heavy rain or when the amount of disease is increasing rapidly.":3,
          "When you notice the infected areas, the first thing you can do is let the plants air out. If they are being grown in a green house, air exposure is a must, because the humidity that the fungus needs to survive is dried up. If the tomatoes are being grown outside, try to keep from wetting the leaves when you are watering the plants. When you water the plants, earlier watering is optimal because it allows the plant time to dry when the sun comes, keeping humidity around leaves to a minimum. Another option for treatment is fungacide sprays.":4,
          "Remove diseased leaves. Improve air circulation. Mulch around the base of the plants. Do not use overhead watering. Control weeds. Use crop rotation. Use fungicidal sprays.":5,
          "Knock mites off plants with water. Use chemical insecticides or miticides.":6,
          "Remove old plant debris at the end of the growing season, rotate crops, don't plant tomatoes in areas where other disease-pror plants have been located in the past year. Pay attention to air circulation. Water tomato plants in the morning so the leaves have time to dry.":7,
          "There is no cure for tomato mosaic virus. Once a plant is infected, it stays that way, and the virus can easily be transmitted to other plants.":8,
          "Use only virus-and whitefly-free tomato and pepper transplants. Transplants should be treated with Capture (bifenthrin) or Venom (dinotefuran) for whitefly adults and Oberon for eggs and nymphs. Imidacloprid or thiamethoxam should be used in transplant houses at least seven days before shipping. Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers. Sanitation is very important for preventing the migration of whitefly adults and the spread of TYLCV. Rogue tomato or pepper plants with early symptoms of TYLCV can be removed from fields by placing infected-looking plants in plastic bags immediately at the beginning season, especially during first three to four weeks. Maintain good weed control in the field and surrounding areas.":9}

tomCause = {"This disease is a result of the bacteria xanthomonas campestris pv. vesicatoria, which can exist on tomato seeds as well as on specific weeds. Bacterial spot can be spread through rain, irrigation, or wet plants. Tomato plants can be infected through pores and wounds, especially in warm, moist weather.":0,
          "early blight is caused by the fungus Alternaria solani.":1,
          "":2,
          "Late blight is caused by the fungus Phytophthora infestans.":3,
          "Tomato leaf mold is a common fungus that attacks tomatoes grown in humid places. The mold attaches itself to the leaf and feeds off of the humidity to grow and thrive.":4,
          "Septoria leaf spot is caused by a fungus, Septoria lycopersici.":5,
          "A heavy spider mite infestation (Acari) caused the curling, distortion, yellow patches and bronzed patches on the leaves of this tomato plant (Lycopersicon).":6,
          "Target spot of tomato is caused by the fungus Corynespora cassiicola":7,
          "There are many ways that a tomato plant can be contaminated with mosaic virus. A common method of infection is by the debris of virally infected plants still in the soil. This virus can survive for at least 50 years in dead, dried plant debris.":8,
          "Tomato yellow leaf curl is a disease of tomato caused by Tomato yellow leaf curl virus.":9}

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
    # serialize model to JSON
import os
import tensorflow as tf
    # Model JSON berisi arsitektur dari Deep learning (Format JSON)


print("Loading Graph...")
global graph

#load model jagung

json_file = open('model/jagung/vggnet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json)
    # load weights into new model
loaded_model2.load_weights("model/jagung/vggnet.h5")
print("Loading Model Corn...")
graph = tf.get_default_graph()
model2 = loaded_model2
print("Loading Model Corn Successful")




    # Load image
import scipy.misc as sm
import numpy as np



def res_jagung():
    
    # print(dic1)
    corDis2 = {}
    for i in corDis:
        corDis2[corDis[i]] = i

    corSym2 = {}
    for i in corSym:
        corSym2[corSym[i]] = i

    corTreat2 = {}
    for i in corTreat:
        corTreat2[corTreat[i]] = i

    corCause2 = {}
    for i in corCause:
        corCause2[corCause[i]] = i

    

    ## Membaca data dan melakukan prediksi dengan model yang sudah di load
    img_path = 'static/jagung2.jpg'
    # im1=np.asarray(sm.imread(img_path))
    img = cv2.imread(img_path)
    img_res = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    im1 = img_res.reshape((1, img_res.shape[0], img_res.shape[1], img_res.shape[2]))
    im1 = im1 / 255.0

    with graph.as_default():
        # from keras.applications.imagenet_utils import decode_predictions
        probs = model2.predict(im1)  # Hitung prob masuk ke setiap kelas

    # prediction = probs.argmax(axis=1)
    prediction = np.argmax(probs)
    print('Prediksi:', corDis2[int(prediction)])
    print(probs)
    print(prediction)
    print(np.amax(probs))
    probability = '{:.4f}'.format(float(np.amax(probs)))
    print('probabilitas : ', probability)
    print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
    # xx=(np.argmax(model.predict(im1)))
    # print (dic1[int(xx)])

    # decode_predictions(preds, 1)
    cv2.destroyAllWindows()
    

    path = 'D:\PKL\PKL5\app\static'
    cv2.imwrite(os.path.join(path, 'keren.jpg'), img)
    cv2.imwrite("static/hJagung.jpg", img)


res_jagung()



