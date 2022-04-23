---
layout: post
title: 'World of Warcraft: The Machine Learning Approach'
---

<style type="text/css">
    .centerImage
    {
        text-align:center;
        display:block;
    }
</style>


As silly as this sounds my first blog post being about World of Warcraft is precious to me. This game is a major reason why I went into mathematics and an activity that drew me closer to family, so to come back to it now with some technical input is a pleasure :)

### Quick Introduction

World of Warcraft is a [MMORPG](https://en.wikipedia.org/wiki/Massively_multiplayer_online_role-playing_game) where players must create the character they play. They choose a faction, race, class, then a specilization of that class. Once you create this character your adventure begins. This freedom to be a tailored individual is exemplified by the interaction between other players. When you team up your individuality contributes to helping the group. While adventuring alone in the open world you must find wepons and quests specific to the identity you chose. It truly is a great time. 

### Meta

The wonderment that World of Warcraft had captured predates its infection. Starting in 2004 WoW came out a year before Youtube and Reddit, two years before its most popular fourm [Wowhead](https://www.wowhead.com), and six years before its encyclopedia [Wowpedia](https://wowpedia.fandom.com). On all of these online resources to the game players read and learn about all the secrets, exploits and meta strategies. This network of information introduced a social pressure to the game generally refered to as "the meta". The meta is discovered when players team up to optimize and expedite every aspect of the game.

This is where the problem begins. The World of Warcraft community has become a [particle swarm optimizer](https://ieeexplore.ieee.org/abstract/document/488968?casa_token=cCQ89OABJjAAAAAA:o4BQOajwvtD5GOR983JxuTebVdruDvjvqlJTIsEw9KU_fm-dRA3Me7_b0z5XJPBICIGo7qmylQ) expediting a game that has a monthly subscription. 

### Thesis
World of Warcraft's (and most other MMORPGs) over developed meta introduces social restrictions to the game that wastes developer time, minimizes membership revenue and magnifies player inequalities.

### The Solution
So how does a game developer make a fun game, immunize it from community swarm optimization without restricting content creators, and protect its monthly subscription?

Our effort will be to create dynamic or intelligent raid bosses. Meaning, that as the community discusses strategies and approaches to beat the boss the boss must do the computational equivalent. To further the challenge we will do this change with no identity change of the current form of World of Warcraft. Here is our plan:
* Introduce a low rank vector space that defines a player's stats
* Analyze a KMeans clustering of this vector space using a concentration algorithm
* Play with the KMeans centroids and randomly sample within a n-sphere to prevent counter analytics that may occur

<img src="https://latex.codecogs.com/svg.image?\block&space;x^2&space;&plus;&space;y^2&space;=&space;z^2" title="https://latex.codecogs.com/svg.image?\block x^2 + y^2 = z^2" class="centerImage">



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
