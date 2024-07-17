from transformers import AutoTokenizer, AutoModelForCausalLM
if __name__ == '__main__':
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",token="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")
    input_text = ('Passages:"<s> .  It is one of the most iconic images of Henry VIII and is one of the most famous '
                  'portraits of any English or British monarch. It was created in 15361537 as part of the \n, '
                  'Westminster, which was destroyed by fire in 1698, but is still well known through many '
                  'copies.\nHans Holbein the Younger, originally from Germany, had been appointed the English King\'s '
                  'Painter in 1536. The portrait was created to adorn the \n. The original mural featured four '
                  'figures arranged around a marble plinth: Henry, his wife \n. The mural was thus commissioned '
                  'sometime during the brief marriage of Henry and Jane Seymour and was completed in 1537. It may '
                  'well have been commissioned to celebrate the coming or actual birth of Henry\'s long-awaited heir, '
                  '\nIt is not clear where in the palace the mural was located, but it may have been in the king\'s '
                  '\nHenry is posed without any of the standard royal accoutrements such as a sword, crown, '
                  'or \n. This was common in progressive royal portraiture of the period, for example the portraits '
                  'by \nof the Habsburg family and other royalty, and also French and German royal portraits.  But '
                  'Holbein\'s success in conveying royal majesty without such specific props is exceptional.  The '
                  'majestic presence is conveyed through Henry\'s aggressive posture, standing proudly erect, '
                  'directly facing the viewer. His legs are spread apart, and arms held from his side in the pose of '
                  'a warrior or a wrestler. In one hand he holds a glove, while the other reaches towards an ornate '
                  'dagger hanging at his waist. Henry\'s clothes and surroundings are ornate, with the original '
                  'painting using gold leaf to highlight the opulence. The detailed \nembroidery is especially '
                  'notable. He wears an array of jewellery including several large rings and a pair of necklaces. His '
                  'large \nand heavily padded shoulders further enhance the aggressive masculinity of the image.\n, '
                  'designed to enhance Henry\'s majesty. It deliberately skews his figure to make him more imposing. '
                  'Comparisons of surviving sets of Henry\'s armour show that his legs were much shorter in reality '
                  'than in the painting. The painting also shows Henry as young and full of health, when in truth he '
                  'was in his forties and had been badly injured earlier in the year in a \naccident. He was also '
                  'already suffering from the health problems that would affect the latter part of his life.\nHenry '
                  'recognized the power of the image Holbein created, and encouraged other artists to copy the '
                  'painting and distributed the various versions around the realm, giving them as gifts to friends '
                  'and ambassadors. Major nobles would commission their own copies of the painting to show their '
                  'loyalty to Henry. The many copies made of the portrait explain why it has become such an iconic '
                  'image, even after the destruction of the original when Whitehall Palace was consumed by fire in '
                  '1698. It has had a lasting effect on Henry\'s public image. For instance, \ndone by Holbein in '
                  'preparation for the portrait group survives in the collection of the \n, showing only the '
                  'left-hand third of the group, with the two Henries.  This was used to make an outline of the '
                  'design on the wall, by pricking holes along the main lines and pushing powdered \nThe cartoon '
                  'differs slightly from the final version. Most notably it shows Henry standing in a more '
                  'traditional three-quarters view rather than the final and iconic head-on position. \nAlso '
                  'surviving is a much smaller half-length portrait of Henry by Holbein that is today in the '
                  'collection of the \nin Madrid. This, the only surviving painting of Henry from Holbein\'s hand, '
                  'may also have been a preparatory study. In it Henry wears much the same clothing as the final '
                  'mural but is still posed in a three-quarters view. For many years this painting was owned by the '
                  '\nAll the remaining copies of the painting are today attributed to other artists, though in most '
                  'cases the name of the copyist is unknown. They vary dramatically in their quality and faithfulness '
                  'to the original source. Most of the reproductions only copy the image of Henry, though a copy by '
                  '\nThe highest quality, and best known, copy is that currently in the collection of the \nBy Hans '
                  'Eworth, oil on 5 oak planks, 229.6 x 124.1cm on display in Trinity College\'s Hall. It was '
                  'commissioned and bequeathed in 1567 by Robert Beaumont, one of the first Masters of the '
                  'college.\n"Tudor and Stuart Portraits From The Collections of the English Nobility and their Great '
                  'Country Houses"\n\n\nPlease read the rules before participating, as we remove all comments which '
                  'break the rules.\nDid a famous painting of Henry VIII holding a turkey leg ever exist? If not, '
                  'what set the precedent for this popular depiction? \nMany popular contemporary depictions of King '
                  'Henry VIII show him holding a turkey leg in one '
                  'hand.\nhttp://40.media.tumblr.com/f11b6990b352ffcdc4aaaa4c17223762/tumblr_nlhf66vNNg1qzstw9o2_540'
                  '.jpg\nSo I went to the Renaissance Fair today and while eating my turkey leg I got to thinking '
                  'about Henry the Eighth. You see a picture of that guy, chances are you\'re looking at a picture of '
                  'a guy with a turkey leg in his hand.\nTurkey drumstick in one hand, lady parts in the other -- '
                  'that\'s how we like our H8.\nI myself believe I have a memory of seeing a renaissance era painting '
                  'of Henry VIII holding a turkey leg in one hand. Many other people believe they remember seeing '
                  'such a portrait as well. (Link to article on \nwhich contains a scene where he devours a chicken, '
                  'but that is a whole chicken, not a singular massive turkey leg.\nAs an aside, some people have '
                  'tried to allege that turkeys did not exist in England during Henry the VIII\'s reign. At least, '
                  'if \nHenry VIII was the first English king to enjoy turkey, although Edward VII made eating turkey '
                  'fashionable at Christmas. It replaced peacocks on the table in Royal Courts\nTL;DR: Does there '
                  'exist a renaissance era painting depicting Henry VIII holding a turkey leg, and if not, '
                  'where does this popular depiction come from?\n\n\nThe Mandela Effect is when a large group of '
                  'people share a common memory of something that differs from what is generally accepted to be '
                  'fact.\na popular Mandela Effect memory that reportedly many people recall: the famous painting of '
                  'King Henry VIII holding a turkey leg in one hand.\nI have a distinct memory of seeing the painting '
                  'in a book when I was a child, and not even recognizing that the object in his hand was a turkey '
                  'leg. I thought it was some kind of club-like weapon. I remember looking at the picture with my '
                  'mom, and her correcting me that it wasn\'t a club, it was a turkey leg.\nThere\'s certainly enough '
                  'evidence that the popular notion of Henry VIII is associated with an image of him holding a turkey '
                  'leg. I (and others) could have swore there was a renaissance era painting depicting him with a '
                  'turkey leg. But searching Google turns up nothing of the sort.\n\n\nThis portrait of the king was '
                  'probably produced by an English or Netherlandish artist working in England. The frame is original '
                  'to the picture, although it has been repainted many times.  Investigation of the paint layers '
                  'indicates that the frame was quite brightly coloured originally, perhaps in imitation of marble or '
                  'tortoiseshell.\n, 2014 (accompanying the exhibition at the National Portrait Gallery from 12th '
                  'September 2014 to 1st March 2015)\nThe Art of the Picture Frame: Artists, Patrons and the Framing '
                  'of Portraits in Britain\n, 1997 (accompanying the exhibition at the National Portrait Gallery from '
                  '8 November 1996 - 9 February 1997), p. 150 \nPainted and gilt oak, mitred lap joint with dowels, '
                  'the top 9 inches of the back of the frame and the corresponding area of the back of the panel '
                  'painted over with a yellow orpiment and white chalk mixture probably in the sixteenth or '
                  'seventeenth centuries, possible traces of scribe lines at one corner on the reverse of the frame, '
                  'two partly filled hanging holes on the reverse at top centre, further holes at top left and top '
                  'right, probably recent but possibly associated with old fixings such as for a picture curtain. 2 '
                  '\n), this is an engaged frame made of oak, held together by corner dowels, although the original '
                  'retaining grooves holding the panel painting in place have long since been cut away at the back. '
                  'The portrait is from an unidentified Anglo-Flemish workshop and appears to have been painted in '
                  'the frame, judging from the ridges of paint at the edge of the portrait. This has been confirmed '
                  'by chemical analysis which shows splashes of the white underpaint used on the panel occurring on '
                  'the extreme inner edge of the frame.\nThe same analysis suggests that the frame has been decorated '
                  'five times. The original finish was a combination of gilding and what appears to be a type of '
                  'graining. At a later date the graining was varnished black, leaving the gilding intact. The whole '
                  'frame was subsequently twice varnished black all over. The frame was probably given its present '
                  'graining and gilding in the nineteenth century.\nWhat was the frame like originally? The areas '
                  'that are gold today were originally gilded. The main part of the frame had a semi-translucent dark '
                  'layer of paint laid directly over a white ground. It is uncertain precisely how this layer would '
                  'have been painted, but it may have been a form of graining, or possibly tortoiseshell or marbling. '
                  'It is made up of a mixture of azurite blue and fine-particled red and brown ochres. The whole '
                  'frame including the gilding was finally given a clear varnish. The picture was probably hung by a '
                  'ribbon or cord through holes pierced through the back and top of the frame.\nA pigment analysis '
                  'was undertaken in 1996 by UCL Painting Analysis Ltd, report no.c891.\nThe former Lord Chancellor '
                  'Sir Thomas More is beheaded for High Treason after refusing to recognise King Henry VIII\'s '
                  'religious supremacy.\nThe French navigator Jacques Cartier explores the sites that will become '
                  'Quebec and Montreal. He names the territory Canada. \nThe Holy Roman Emperor Charles V captures '
                  'the city of Tunis in present-day Tunisia from the Ottoman admiral Khair ad-Din, called Barbarossa. '
                  '\nWe are currently unable to accept new comments, but any past comments are available to read '
                  'below.\n. Please note that we cannot provide valuations. You can buy a print or greeting card of '
                  'most illustrated portraits. Select the portrait of interest to you, then look out for a \nbutton. '
                  'Prices start at around 6 for unframed prints, 16 for framed prints. If you wish to license an '
                  'image, select the portrait of interest to you, then look out for a \n. We digitise over 8,'
                  '000 portraits a year and we cannot guarantee being able to digitise images that are not already '
                  'scheduled.\nJoin our newsletter and follow us on our social media channels to find out more about '
                  'exhibitions, events and the people and portraits in our Collection.\n\n\nWhat is King Henry the '
                  'Eighth holding in his hand in this painting? - Factual Questions - Straight Dope Message '
                  'Board\nIts probably the most famous depiction of him. Yet I have never heard anyone talk about '
                  'that thing in his hand.\npictures painted of himself holding his gloves? Was holding gloves a big '
                  'thing in Tudor times?\nwas a notorious womanizer; he had no fewer than seven illegitimate '
                  'children. (He also used to eat so much that he would throw up in order to keep eating.)\non the '
                  'other hand, couldnt even get it up. Deformed and retarded, he was the end result of generations of '
                  'Hapsburg inbreeding.\nwas a notorious womanizer; he had no fewer than seven illegitimate children. '
                  '(He also used to eat so much that he would throw up in order to keep eating.)\nwas a notorious '
                  'womanizer; he had no fewer than seven illegitimate children. (He also used to eat so much that he '
                  'would throw up in order to keep eating.)\nCompared to these mighty sperm banks, Edward IV must '
                  'perforce weep from inadequacy :p.\n* 1713-1719 with Maria Magdalena of Bielinski, by her first '
                  'marriage Countess of Dnhoff and by the second Princess Lubomirska.\nThere may be some padding (due '
                  'to fabric and/or artistic licence). But many of the contemporary commentators noted the impressive '
                  'physique of Henry, when he was in his prime.\nYes, holding your gloves became a standard detail in '
                  'Renaissance/Baroque portraiture. Status symbol/sign of refinement.\nThere may be some padding (due '
                  'to fabric and/or artistic licence). But many of the contemporary commentators noted the impressive '
                  'physique of Henry, when he was in his prime.\nClose-fitting combat dress at anniversary display '
                  'shows Henry VIII\'s ballooning figure\nA friend of mine had a large format poster poster of his '
                  'last suit of armour.  It was quite impressive.  I would like to see one that has his various suits '
                  'in order showing the progress.\nNo, he wasnt. Henry VIIIs grandfather was Edmund Tudor, '
                  'Earl of Richmond.  Edward IV was the older brother of Richard III, who Henry VIIIs father, '
                  'Henry Tudor (later VII), deposed at the Battle of Bosworth.  Edward IV had no (legitimate) '
                  'grandchildren due to his brother (probably) offing his only two legitimate sons (the Princes in '
                  'the Tower and all that jazz).\nThe Tudor claim to the throne derived from John of Gaunt, '
                  'third son of Edward III, and Henry VIIIs great-great-great-grandfather.\nYes, holding your gloves '
                  'became a standard detail in Renaissance/Baroque portraiture. Status symbol/sign of refinement.\n, '
                  'daughter of Edward IV.  Her second son by Henry VII was Henry, Duke of York, who succeeded to the '
                  'throne as Henry VIII.\n\n\nThis full length portrait of the English monarch King Henry VIII is '
                  'derived from the Whitehall Mural, painted by Hans Holbein in 1537. It is one of the most '
                  'recognisable images in the Walker Art Gallerys collection. The mural depicted Henry VIII, '
                  'Henry VII and their wives, Jane Seymour and Elizabeth of York respectively. It was painted onto '
                  'the wall of one of the state rooms of Whitehall Palace. It was probably intended to serve as '
                  'propaganda to reinforce the strength of the Tudor dynasty and Henry VIIIs total authority. It was '
                  'destroyed in a fire at Whitehall Palace in 1698. The Walker portrait was produced by an unknown '
                  'artist who was familiar with the Whitehall mural. The artist had access to"\nQuestion: What is '
                  'King Henry holding in the Portrait of Henry VIII?\nPlease find the answer to the question from the '
                  'above passages and generate the answer text. If there is an answer in the documents, please keep '
                  'the answer authentic to the passage, if the question is to ask for opinion or if there is no '
                  'answer found in the documents, please output "I have no comment".\nAnswer:\n')
    messages = [{"role": "user", "content": input_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    print(len(inputs[0]))
    out = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model_output = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    #model_output = model_output[len(input_text):]
    model_output = model_output.split("[/INST]")[1]
    print("-------------------------------------------------------")
    print(model_output, flush=True)