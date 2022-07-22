---
layout: post
title: 'World of Warcraft: The Machine Learning Approach'
---

Lot’s of gamers that were brought up in my generation are questioning if games are still fun or if their enjoyment is nostalgia that we are trying to relive. Today’s oeuvre is this blog’s first, and a specific solution to an observation I have had that I am going to name cybersocial optimization. 

I think that gamers are experiencing cybersocial optimization of their gaming experience, this post will be a technical solution to this social phenomenon.

The game that this solution is intended for is for the legendary World of Warcraft released in 2004, but the philosophy behind it could be applied other places.

Let’s talk about World of Warcraft and its cybersocial situation. World of Warcraft was released as an enormous place for players to explore, fight, make friends, and team up to defeat enemies that stood 50 feet tall. The world was detailed, you would spend years learning the lore from the variety of different races of creatures all with your friends you had made over the internet. For some, this may have been the first time where they met a true friend over the internet. This popularity spun out of control. Millions of players from all of the world joined together to defeat enemies of The Alliance or Horde to collect powerful weapons and armor. The collection of this arsenal took months, guilds of players were created to defeat the raid bosses. Which took, in many cases, hundreds of tries consisting of very coordinated efforts of dozens of players representing months of hard work just to defeat one. This is when games were fun, or at least, this is the time where many gamers remember their favorite games being fun.

Back then, very small internet forums were made for the most dedicated players to share secrets about how to have the best try at defeating a raid boss. Little gems of advice found in the nooks of the internet. Later, social media created standard places for the community to interact, and boy did the community interact.

Now there are data science dashboards that show your live performance of your character, simulations that show you exactly which spells to cast in real time, data miners get gameplay clues about patches and expansions that haven’t even been released, and youtube creators race to see who will make the best boss fight guide. Games in the past weren’t more fun, they just weren’t being played for you.

How can foes in World of Warcraft do equivalent cybersocial optimization about the gamers? If this were possible the community could grow and interact as much as they would like and the game would be immunized from them.

### The Old Stats

In World of Warcraft, to become more powerful, every player goes on a quest to gain more powerful weapons and more protective armor. Players have a choice of a class (Mage, Warrior, Paladin, etc…) and each class has multiple specializations but all of their goals remain the same. Get the best items, which give the best stats, and use them to get the next best items and stats – until you are ready to fight the raid bosses.

In combat of world of warcraft a player utilizes the armor and weapons they have obtained in order to survive. These statistics from their gear help them do more damage and protect themselves from damage, however, the arsenal of spells each player character has allows them to augment their ability to do damage and protect themselves. For the sake of this post we will call a players ability to do damage DPS (damage per second) and their ability to protect themselves Defense Ability. DPS and Defense Ability are postive real number values that are determined by the players skill and the stats on their armor.

<figure>
<img src="\public\stats.png" alt="Base Stats" class="center">
<figcaption>A representation of a character with stats shown</figcaption>
</figure>

Player's stats are restricted by their class. A mage will never have a high strength stat and a mage player needs to rely on a warrior player in order to deal with close hand to hand combat. This inspires comradery and thus we can't mess with this od stat line to much. If we did, we could ruin the delicate balance between classes and their need for each other. So, the old stays.

### New Stat Line
World of Warcraft is big. Sometimes you’re fighting in a place called the Firelands, and sometimes you’re underwater fighting the Naga. All have beautiful artistic themes and the weapons and armor that drop there all have some themed visual presentation. Let’s put this geographical theme to use. 

Fire, Frost, Holy, Shadow, Arcane, Nature are a pretty good set of themes that players fight in, but also are the themes of their classes. Paladins and Priests are students of the light, Mages study fire, frost, and the arcane. Druid’s fight with the power of nature, there are more classes but you get the point.

Additionally to the DPS and armor rating of a player, let’s add another vector space that represents what themed armor the player has on. Called the Elemental Vector Space. Each vector in the elemental vector space will be normalized and summing across the elements of the vector results in one. Each basis vector in this space will represent the elements mentioned previously.

<figure>
<img src="\public\elemental_stats_comp.jpeg" alt="More Stats" class="center">
<figcaption>Side by side of suggested stats</figcaption>
</figure>

To frame our testing later we will assign two elemental vectors to each player. One for their currently equipped weapons and one for their equipped armor respectively named "elemental attack" and "elemental defense". If we multiply the DPS and Defensive ability by their corresponding elemental vectors we are given a vector that represents the distribution of DPS or Defensive ability across the different elements in the game. We will call these vectors the "total attack" and "total defense" vectors and they are the ones that we will focus on for the rest of the paper:

<p><span class="math display">\[total\attack = DPS * elemental\attack\]</span></p>
<p><span class="math display">\[total\defense = Defense\Ability * elemental\defense\]</span></p>

This way, a player who is a warrior loves the fire themed areas then their characters swords would glow with fire dealing more fire DPS or a paladin enjoyed the theme of the crypts and the undead and now their prayers worship a shadow demon.

Alright pre-algorithm round up:
1. We have talked about DPS and Defensive Ability which are positive real values that grow as players level up and get better armor and improve their skills at the game. 
2. We have suggested an additional vector space called the "Elemental Vector Space" that gives us an idea from which elemntal power weapons and armor come from.
3. Multiplying DPS/Defensive Ability and the elemental attack/defense vectors give us a vector showing the type of DPS or type of defensive power each player has. This vectors are called "total attack/defense vectors" and these vectors are what we will be focusing on for the algorithm testing.


### Developing the algorithm

To get the boss to counter cybersocial optimization we want to make a crude simulation of what an encounter with a raid boss is for players. Adding the attack vector of players together we get a raid. A raid and a boss will exchange blows and whoever takes more turns to kill the other loses. 

A raid will attack with its total attack vector and the boss will defend with its total defense vector. The two vectors are subtracted elementwise and any positive value remaining is summed together to represent the total damage done to the boss every turn, vise versa. 

To obtain a raid’s collective attack and defense vector we will generate a random attack/defense elemental vectors, where elements sum together to be one and multiply it by the raid’s collective DPS and Defensive Ability resulting in total attack and total defense vectors

To show cybersocial optimization we will initialize 200 raids to fight the boss and add a bias term to half of the vectors, creating a cluster representing those who are implementing some dominant strategy on the many internet forums. After a PCA this is what it looks like.

<figure>
<img src="\public\raid_attack_cluster.png" alt="a cluster" class="center">
</figure>

As you can see the cybersocial optimization is prevalent where the tightest concentration of raids are. Before we do any machine learning lets run this random initialization of the raids against a random initialization of the boss.

When the boss has done no fitting we will track two scenarios, when the boss random initialization favors the boss and when the random initialization favors the raids, so here is the score.

<b>Random initialization favors boss: Boss Score: 1300 Raid Score: 700</b>
<b>Random initialization favors raids: Boss Score: 855 Raid Score: 1145</b>

In the scatter plot shown above we see black centroids, derived from the KMeans algorithm, for each group we can calculate which group is the most concentrated. That group represents some sort of cybersocial optimization of an attack vector. If the boss sets that centroid as its defense vector it would hopefully maximize the mitigated damage incoming from the raid. Checking the scores we see these changes:

<b>Random initialization favors boss: Boss Score: 1031 Raid Score: 969</b>
<b>Random initialization favors raids: Boss Score: 1022 Raid Score: 978</b>

The boss is winning the majority of the battles against the raids in both cases. We will fit the bosses attack vector along with its defense vector using the same method.

<b>Random initialization favors boss: Boss Score: 1004 Raid Score: 996</b>
<b>Random initialization favors raids: Boss Score: 1008 Raid Score: 992</b>

Why does this number even out the more KMeans fitting we do? In our experiment we made 200 raids, 100 random ones and 100 biased ones representing the cybersocial optimized ones. So the boss is killing the optimized raids and being killed by those not following the cybersocial optimization. 

New problem. If this game play mechanic was introduced to the game then most players would not have any interaction with it at all. If you are unaware of the cybersocial optimization of the game then all this complicated machine learning means nothing. There is another problem, World of Warcraft has many API’s that allow developers to make add-ons and analyze in-game statistics. If players who were serious about the cybersocial optimization create an add-on then they could know what the boss would fit next.

So we need to create some variance that scrambles the attempts to make counter analytics and includes more players than just the ones that chase cybersocial optimization. To do this we will set the centroid of the KMeans cluster that is the most concentrated as the center of an n-sphere. We can generate Cartesian coordinates from the following:

<p><span class="math display">\[x_1 = r cos(\phi_1 )\]</span></p>
<p><span class="math display">\[x_2 = r sin(\phi_1 ) cos(\phi_2 )\]</span></p>
<p><span class="math display">\[x_3 = r sin(\phi_1 ) sin(\phi_2 ) cos(\phi_3 )\]</span></p>
<p><span class="math display">\[\dots\]</span></p>
<p><span class="math display">\[x_{n-1} = r sin(\phi_1 ) \dots sin(\phi_{n-2} ) cos(\phi_{n-1} )\]</span></p>
<p><span class="math display">\[x_{n} = r sin(\phi_1 ) \dots sin(\phi_{n-2} ) sin(\phi_{n-1} )\]</span></p>

Once this n-sphere is generated we can sample a new total attack and total defense vector from with in it. Assigning those n-sphere'd sampled vectors and assigning them to the raid boss we have this turn out:

<b>Random initialization favors boss: Boss Score: 1324 Raid Score: 676</b>
<b>Random initialization favors raids: Boss Score: 1611 Raid Score: 389</b>

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
