---
layout: post
title: 'World of Warcraft: The Machine Learning Approach'
thumbnail: "_posts\thumb.jpg"
---

<style type="text/css">
    .centerImage
    {
        text-align:center;
        display:block;
    }
</style>



Lot’s of gamers that were brought up in my generation are questioning if games are still fun or if their enjoyment is nostalgia that we are trying to relive. Today’s oeuvre is this blog’s first, and a specific solution to an observation I have had that I am going to name cybersocial optimization. 

<p><span class="math display">\[y = \frac{a}{b} + c^2 + d\]</span></p>

I think that gamers are experiencing cybersocial optimization of their gaming experience, this post will be a technical solution to this social phenomenon.

The game that this solution is intended for is for the legendary World of Warcraft released in 2004, but the philosophy behind it could be applied other places.

Let’s talk about World of Warcraft and its cybersocial situation. World of Warcraft was released as an enormous place for players to explore, fight, make friends, and team up to defeat enemies that stood 50 feet tall. The world was detailed, you would spend years learning the lore from the variety of different races of creatures all with your friends you had made over the internet. For some, this may have been the first time where they met a true friend over the internet. This popularity spun out of control. Millions of players from all of the world joined together to defeat enemies of The Alliance or Horde to collect powerful weapons and armor. The collection of this arsenal took months, guilds of players were created to defeat the raid bosses. Which took, in many cases, hundreds of tries consisting of very coordinated efforts of dozens of players representing months of hard work just to defeat one. This is when games were fun, or at least, this is the time where many gamers remember their favorite games being fun.

Back then, very small internet forums were made for the most dedicated players to share secrets about how to have the best try at defeating a raid boss. Little gems of advice found in the nooks of the internet. Later, social media created standard places for the community to interact, and boy did the community interact.

Now there are data science dashboards that show your live performance of your character, simulations that show you exactly which spells to cast in real time, data miners get gameplay clues about patches and expansions that haven’t even been released, and youtube creators race to see who will make the best boss fight guide. Games in the past weren’t more fun, they just weren’t being played for you.

How can foes in World of Warcraft do equivalent cybersocial optimization about the gamers? If this were possible the community could grow and interact as much as they would like and the game would be immunized from them.

# Let’s Make Some Data

In World of Warcraft, to become more powerful, every player goes on a quest to gain more powerful weapons and more protective armor. Players have a choice of a class (Mage, Warrior, Paladin, etc…) and each class has multiple specializations but all of their goals remain the same. Get the best items, which give the best stats, and use them to get the next best items and stats – until you are ready to fight the raid bosses.


In combat of world of warcraft a player utilizes the armor and weapons they have obtained in order to survive. These statistics from their gear help them do more damage and protect themselves from damage, however, the arsenal of spells each player character has allows them to augment their ability to do damage and protect themselves.

Player’s stats are important to the lore of the game, since identity of the player's classes is based on their stats, changing these on a whim, or equalizing them, would make the choices players make in their character creation pointless. We really can’t be creative and change things up here because then everybody can be anything and effectively nothing.

For each class different stats are necessary to perform. A Mage’s fireball spell is not based on the strength stat, in fact none of the Mage abilities use the strength stat so you can imagine a mage won’t look for items with strength. So you have all of these stats, and each class has a set of functions (spells) that use these stats as parameters to figure out how much damage you do to your enemies. This damage output is usually measured by the community as a player's DPS (damage per second). DPS is a more true measurement of player skill, because it values both player power from weapons and armor, but then their ability to cast the right spells at the right time. DPS has a twin analytic more about defense, we will call it Defense Ability. Defense Ability like DPS is based on the armor that a player is wearing but also the players ability to cast spells to help mitigate damage.

Back to the World of Warcraft. It’s big. Sometimes you’re fighting in a place called the Firelands, and sometimes you’re underwater fighting the Naga. All have beautiful artistic themes and the weapons and armor that drop there all have some themed visual presentation. Let’s put this geographical theme to use. 

Fire, Frost, Holy, Shadow, Arcane, Nature are a pretty good set of themes that players fight in, but also are the themes of their classes. Paladins and Priests are students of the light, Mages study fire, frost, and the arcane. Druid’s fight with the power of nature, there are more classes but you get the point.

[PICTURE OF THE SIDE BY SIDE STATS]

Additionally to the DPS and armor rating of a player, let’s add another vector space that represents what themed armor the player has on. Called the Elemental Vector Space. Each vector in the elemental vector space will be normalized and summing across the elements of the vector results in one. Every player will have two elemental vectors, one based on their armor called the defense vector and one for the DPS called the attack vector considering the elemental theme of the weapons a player has. To calculate these vectors we will multiply the DPS against the attack elemental vector and the armor rating against the defense elemental vector outputting a distribution of DPS and armor rating across the elements listed previously.

[DPS * vector equations]


This way, a player who is a warrior loves the fire themed areas then their characters swords would glow with fire dealing more fire DPS or a paladin enjoyed the theme of the crypts and the undead and now their prayers worship a shadow demon.

Alright pre-algorithm round up:
We have talked about DPS and armor rating which are scalar values that grow as players level up and get better armor and improve their skills at the game.
We have introduced the elemental vector space which is a statistical representation of where the players have been adventuring. Giving their DPS and armor that they improve over their adventure an “elemental theme” based on the armor they are currently wearing.
Multiplying the elemental vector and the DPS scalar gives us a vector where every element of the vector is the portion of the total DPS that a player does with that element. Same with the armor rating scalar. 
We will call the resultant vector of the DPS multiplied by the elemental vector the ‘attack vector’ and the armor rating scalar multiplied by the elemental vector the ‘defense vector’.

Developing the algorithm

To get the boss to counter cybersocial optimization we want to make a crude simulation of what an encounter with a raid boss is for players. Adding the attack vector of players together we get a raid. A raid and a boss will exchange blows and whoever takes more turns to kill the other loses. A raid will attack with its attack vector and the boss will defend with its defense vector. The two vectors are subtracted elementwise and any positive value remaining is summed together to represent the total damage done to the boss every turn, vise versa. To obtain a raid’s collective attack and defense vector we will generate a random vector, where elements sum together to be one and multiply it by the raid’s collective DPS and armor rating.

To show cybersocial optimization we will initialize 200 raids to fight the boss and add a bias term to half of the vectors, creating a cluster  representing those who are implementing some dominant strategy on the many internet forums. After a PCA this is what it looks like.

[SHOW 2D VIEW]

As you can see the cybersocial optimization is prevalent where the tightest concentration of raids are. Before we do any machine learning lets run this random initialization of the raids against a random initialization of the boss.

When the boss has done no fitting we will track two scenarios, when the boss random initialization favors the boss and when the random initialization favors the raids, so here is the score.

Random initialization favors boss: Boss Score: 1300 Raid Score: 700
Random initialization favors raids: Boss Score: 855 Raid Score: 1145

In the scatter plot shown above we see black centroids, derived from the KMeans algorithm, for each group we can calculate which group is the most concentrated. That group represents some sort of cybersocial optimization of an attack vector. If the boss sets that centroid as its defense vector it would hopefully maximize the mitigated damage incoming from the raid. Checking the scores we see these changes:

Random initialization favors boss: Boss Score: 1031 Raid Score: 969
Random initialization favors raids: Boss Score: 1022 Raid Score: 978

The boss is winning the majority of the battles against the raids in both cases. We will fit the bosses attack vector along with its defense vector using the same method.

Random initialization favors boss: Boss Score: 1004 Raid Score: 996
Random initialization favors raids: Boss Score: 1008 Raid Score: 992

Why does this number even out the more KMeans fitting we do? In our experiment we made 200 raids, 100 random ones and 100 biased ones representing the cybersocial optimized ones. So the boss is killing the optimized raids and being killed by those not following the cybersocial optimization. 

New problem. If this game play mechanic was introduced to the game then most players would not have any interaction with it at all. If you are unaware of the cybersocial optimization of the game then all this complicated machine learning means nothing. There is another problem, World of Warcraft has many API’s that allow developers to make add-ons and analyze in-game statistics. If players who were serious about the cybersocial optimization create an add-on then they could know what the boss would fit next.

So we need to create some variance that scrambles the attempts to make counter analytics and includes more players than just the ones that chase cybersocial optimization. To do this we will set the centroid of the KMeans cluster that is the most concentrated as the center of an n-sphere.

[block math]

Random initialization favors boss: Boss Score: 1324 Raid Score: 676
Random initialization favors raids: Boss Score: 1611 Raid Score: 389

We see that with the sampling from the inside of an n-sphere our boss is able to beat not just the cybersocial optimization but a good amount of the raids not participating in the cybersocial optimization. Which is good, our goal was to beat the cybersocial players and make an engaging gameplay mechanic for everyone. The n-sphere addition completes that goal!

Algorithm Summary
Players explore the world to obtain armor and weapons that reflect the theme of the part of the world they obtained effectively choosing what their elemental stat line looks like.
The KMeans enabled boss clusters the players and discovers a concentrated clustering of players (most likely due to some cybersocial optimization) using newly formed attack and defense vectors derived from the new elemental stat line.
The boss then samples a point within the n-sphere centered at the centroid of the most concentrated cluster and makes that point its new attack and defense vectors.




[Jekyll](http://jekyllrb.com)

### Built on Poole

Poole is the Jekyll Butler, serving as an upstanding and effective foundation for Jekyll themes by [@mdo](https://twitter.com/mdo). Poole, and every theme built on it (like Lanyon here) includes the following:

* Complete Jekyll setup included (layouts, config, [404](/404), [RSS feed](/atom.xml), posts, and [example page](/about))
* Mobile friendly design and development
* Easily scalable text and component sizing with `rem` units in the CSS
* Support for a wide gamut of HTML elements
* Related posts (time-based, because Jekyll) below each post
* Syntax highlighting, courtesy Pygments (the Python-based code snippet highlighter)

### Lanyon features

In addition to the features of Poole, Lanyon adds the following:

* Toggleable sliding sidebar (built with only CSS) via **☰** link in top corner
* Sidebar includes support for textual modules and a dynamically generated navigation with active link support
* Two orientations for content and sidebar, default (left sidebar) and [reverse](https://github.com/poole/lanyon#reverse-layout) (right sidebar), available via `<body>` classes
* [Eight optional color schemes](https://github.com/poole/lanyon#themes), available via `<body>` classes

[Head to the readme](https://github.com/poole/lanyon#readme) to learn more.

### Browser support

Lanyon is by preference a forward-thinking project. In addition to the latest versions of Chrome, Safari (mobile and desktop), and Firefox, it is only compatible with Internet Explorer 9 and above.

### Download

Lanyon is developed on and hosted with GitHub. Head to the <a href="https://github.com/poole/lanyon">GitHub repository</a> for downloads, bug reports, and features requests.

Thanks!
