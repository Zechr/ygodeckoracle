import random
import math
import sys
database = dict()
category_data = [dict() for x in range(6)]
alpha = 0.001
beta = 0.5

def trainPowerVal(textfile):
  turns = textfile.splitlines()
  
  turnbase = dict()
  #there are 10 lines of data per turn of the game
  #the goal of the game is to have more cards on the board and in the hand than the opponent
  #so this classifier section measures the strength of cards based on that
  for i in range(0, (len(turns)/10)*10, 10):
    score = 0.0
    #line 1 has the players' life points at the beginning of the turn
    lp_played = turns[i].split(',')
    #line 2 has the turn player's cards in hand at the beginning of the turn
    your_hand = turns[i+1].split(',')
    #line 3 has the opponent player's cards in hand at the beginning of the turn
    op_hand = turns[i+2].split(',')
    #line 4 has the cards on the turn player's field at the beginning of the turn
    your_field = turns[i+3].split(',')
    #line 5 has the cards on the opponent player's field at the beginning of the turn
    op_field = turns[i+4].split(',')
    #line 6 has the players' life points at the end of the turn
    lp_r = turns[i+5].split(',')
    #line 7 has the turn player's cards in hand at the end of the turn
    your_hand_r = turns[i+6].split(',')
    #line 8 has the opponent player's cards in hand at the end of the turn
    op_hand_r = turns[i+7].split(',')
    #line 9 has the cards on the turn player's field at the end of the turn
    your_field_r = turns[i+8].split(',')
    #line 10 has the cards on the opponent player's field at the end of the turn
    op_field_r = turns[i+9].split(',')
    
    #calculates how much the difference in the two player's life points changed
    lp_dif = (int(lp_r[0]) - int(lp_r[1])) - (int(lp_played[0]) - int(lp_played[1]))
    #calculates how much the difference in the number of cards in each player's hand changed
    hand_dif = (len(your_hand_r) - len(op_hand_r)) - (len(your_hand) - len(op_hand))
    #calculates how much the difference in the number of cards on each player's field changed
    field_dif = (len(your_field_r) - len(op_field_r)) - (len(your_field) - len(op_field))
             
    #weighting system for a card advantage score
    score = field_dif*0.5 + hand_dif*0.3 + lp_dif*0.2/1000
    #line 1 also contains the cards the turn player played that turn, so all those cards will be associated with this turn's score
    for j in range(2, len(lp_played)):
      if database.has_key(lp_played[j]):
        database[lp_played[j]] = database[lp_played[j]] + score
        turnbase[lp_played[j]] = turnbase[lp_played[j]] + 1
      else:
        database[lp_played[j]] = score
        turnbase[lp_played[j]] = 1
  for key in database:
    #divides total score of each card by the number of uses then applies smoothing
    database[key] = abs(database[key]/(1.0*turnbase[key]))
    if database[key] < 1:
      database[key] = database[key] + beta
    
def trainHandVal(textfile):
  """Format: Your LP,OP LP
             YourHandCard1,YourHandCard2,etc
             OPHandCard1,OPHandCard2,etc
             YourFieldCard1,YourFieldCard2
             OPFieldCard1,OPFieldCard2
             Repeat w/ result 10 lines per hand"""
  """Category:
    0, going minus up to going 0.5 (setting and passing)
    1, 0.5 to plus 1
    2, plus 1 to plus 2
    3, plus 2 to 4
    4, plus 4 to 6
    5, plus 6 or more"""
  turns = textfile.splitlines()
  cat_count = [0 for m in range(6)]
  for i in range(0, len(turns), 10):
    score = 0.0
    lp_played = turns[i].split(',')
    your_hand = turns[i+1].split(',')
    op_hand = turns[i+2].split(',')
    your_field = turns[i+3].split(',')
    op_field = turns[i+4].split(',')
    lp_r = turns[i+5].split(',')
    your_hand_r = turns[i+6].split(',')
    op_hand_r = turns[i+7].split(',')
    your_field_r = turns[i+8].split(',')
    op_field_r = turns[i+9].split(',')
    
    #the change in the difference in the two player's life points
    lp_dif = (int(lp_r[0]) - int(lp_r[1])) - (int(lp_played[0]) - int(lp_played[1]))
    
    #calculates the difference in hand advantage using the scores learned from trainPowerVal()
    temp1 = 0.0
    temp2 = 0.0
    for card in your_hand:
      temp1 += database.get(card, 1)
    for card in op_hand:
      temp2 += database.get(card, 1)
    hand_dif = temp1 - temp2
    temp1 = 0.0
    temp2 = 0.0
    for card in your_hand_r:
      temp1 += database.get(card, 1)
    for card in op_hand_r:
      temp2 += database.get(card, 1)
    hand_dif = (temp1 - temp2) - hand_dif
    
    #calculates the difference in field advantage using the scores learned from trainPowerVal()
    temp1 = 0.0
    temp2 = 0.0
    for card in your_field:
      temp1 += database.get(card, 1)
    for card in op_field:
      temp2 += database.get(card, 1)
    field_dif = temp1 - temp2
    temp1 = 0.0
    temp2 = 0.0
    for card in your_field_r:
      temp1 += database.get(card, 1)
    for card in op_field_r:
      temp2 += database.get(card, 1)
    field_dif = (temp1 - temp2) - field_dif
    
    score = 0.2/1000*lp_dif + 0.3*hand_dif + 0.5*field_dif
    
    category = 0
    if score < 0.5:
      category = 0
    elif score < 1:
      category = 1
    elif score < 2:
      category = 2
    elif score < 4:
      category = 3
    elif score < 6:
      category = 4
    else:
      category = 5
    cat_count[category] = cat_count[category] + 1
    #associates each card in the hand with the generatedd score
    for card in your_hand:
      if category_data[category].has_key(card):
        category_data[category][card] = category_data[category][card] + 1
      else:
        category_data[category][card] = 1
  #divides each feature's count for a given classification category by the number of times that category appeared
  #computes reverse probability for bayes law
  for n in range(len(category_data)):
    cat = category_data[n]
    for key in cat:
      cat[key] = cat[key]/(1.0*cat_count[n])

def classify(decklist):
  cat_count = [0 for n in range(6)]
  cards = decklist.splitlines()
  #run 100 randomized trials to get distribution
  for i in range(100):
    cats = [0 for x in range(6)]
    hand = []
    #draw 5 random cards if going first or 6 if going second, assume half and half
    if i%2 == 0:
      hand_index = random.sample(range(0, len(cards)), 5)
      for card_index in hand_index:
        hand.append(cards[card_index])
    else:
      hand_index = random.sample(range(0, len(cards)), 6)
      for card_index in hand_index:
        hand.append(cards[card_index])
    #assume independence of cards, add the log probabilities, then argmax
    for card in hand:
      for k in range(6):
        if category_data[k].has_key(card):
          cats[k] = cats[k] + math.log(category_data[k].get(card))
        else:
          cats[k] = cats[k] + math.log(alpha)
    amax = cats.index(max(cats))
    cat_count[amax] = cat_count[amax] + 1
  for count in cat_count:
    print count

"""turnTrain = open('turnTrain.txt', 'r')
handTrain = open('tellarhandtrain.txt', 'r')
deck = open('TellarDeck.txt', 'r')
trainPowerVal(turnTrain.read())
trainHandVal(handTrain.read())
classify(deck.read())"""

"""Inputs should be in the order of turn training data, hand training data, and test decklist
  Make sure all are pure text files accessible from the directory this is running on"""
def main():
  if (len(sys.argv) < 4):
    print "Turn, hand, and decklist data needed"
    return
  turnTrain = open(sys.argv[1], 'r');
  handTrain = open(sys.argv[2], 'r');
  deck = open(sys.argv[3], 'r');
  trainPowerVal(turnTrain.read())
  trainHandVal(handTrain.read())
  classify(deck.read())

if __name__ == "__main__":
  main()