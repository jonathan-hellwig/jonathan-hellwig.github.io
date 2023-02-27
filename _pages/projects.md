---
layout: page
title: Projects
permalink: /projects/
description: Some of the projects I have worked on in the past.
nav: true
nav_order: 2
display_categories: [work, fun]
horizontal: false
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/robocup_profile.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

# Relations between variants of stochastic gradient descent and stochastic differential equations

In recent years, deep learning has caught the interest of many researchers. The progress in this field can be observed in its applications in robotics, medicine, physics and even in new generative AI technologies like ChatGPT and StableDiffusion. However, while the practical relevance of deep learning is undeniable, some underlying mechanisms and properties of these deep learning models are not well understood from a theory perspective. One particular topic of interest are the dynamics of parameters during the training process of neural networks. Most modern deep neural networks are trained using a variant of the prototypical stochastic gradient descent method (SGD). A recent work by [Li et al.](https://arxiv.org/abs/1511.06251) introduces a time-continuous model for the dynamics of SGD. In my thesis, I set out to investigate the applicability of this model to deep neural network. As a starting point, I used the work by [Li et al.](https://arxiv.org/abs/2102.12470) that introduces an efficient algorithm to simulate this time-continuous model.


Let me present an example to show how we can approximate the dynamics of a discrete optimization process by a continuous process:
For a given set of points $$\{(x_i, y_i)\}_{i=1}^n$$, where $$x_i, y_i \in \mathbb{R}$$ for $$i = 1, \dots, n$$, consider the linear regression problem:

$$
\min_{w,b \in \mathbb{R}}\mathcal{L}(w,b) = \frac{1}{2n}\sum_{i=1}^n\left(w x_i + b - y_i\right)^2
$$

The iterates of the gradient descent algorithm with learning rate $$\eta > 0$$ initial values $$w_{0},b_0 \in \mathbb{R}$$ are given by

$$
\begin{pmatrix}
      w_{k+1} \\
      b_{k+1}
    \end{pmatrix}
    =
    \begin{pmatrix}
      w_{k} \\
      b_k
    \end{pmatrix}
    - \frac{\eta}{n}
    \begin{pmatrix}
      \left(\displaystyle\sum_{i=1}^n x_i^2\right)w_k +  \left(\displaystyle\sum_{i=1}^n x_i\right)b_k -  \displaystyle\sum_{i=1}^n x_i y_i \\
       \left(\displaystyle\sum_{i=1}^n x_i\right) w_k + nb_k -  \displaystyle\sum_{i=1}^n y_i
    \end{pmatrix}.
$$

In the figure below, we can see the trajectory of the weight $$w$$ and bias $$b$$ for different choices of the learning rate $$\eta > 0$$.
<div class="l-page">
  <iframe src="{{ '/assets/plotly/linear_model.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>
If we use a different scaling for the x-axis of the figure and interpret the learning rate $$\eta > 0$$ as a time step, we observe that all trajectories coincide in the next figure.  
<div class="l-page">
  <iframe src="{{ '/assets/plotly/linear_model_scaled.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>
Indeed, as we send the learning rate $$\eta > 0$$ to zero, we obtain a system of ordinary differential equations:

$$
\begin{pmatrix}
      w'(t)      \\
      b'(t)     
  \end{pmatrix}
  = \frac{1}{n}
  \begin{pmatrix}
    -  \displaystyle\sum_{i=1}^n x_i^2  &  -  \displaystyle\sum_{i=1}^n x_i      \\
      -  \displaystyle\sum_{i=1}^n x_i  &  -n      
  \end{pmatrix}
  \begin{pmatrix}
    w(t)     \\
    b(t)    
  \end{pmatrix}
  + \frac{1}{n}
  \begin{pmatrix}
      \displaystyle\sum_{i=1}^n x_i y_i     \\
      \displaystyle\sum_{i=1}^n y_i    
  \end{pmatrix}.
$$

This equation is a special case of the well-known gradient flow equation. For stochastic gradient descent there exists a stochastic generalization in form of a stochastic differential equation. 

If you are interested in a more complete and rigorous introduction to this topic, you can find my thesis, source code and talk slides [at my GitHub page](https://github.com/jonathan-hellwig/master_thesis).
<!-- 
# Object detection in the RoboCup SPL

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/robot.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
In the RoboCup SPL league, ten small humanoid robots compete in teams of five to score goals in a game of football. The robots play completely autonomously and once the game has started the robots are on their own. As a result, the robots need to detect line markings, other robots, the ball and the goal posts in order to plan their next move. Our current implementation allows us to locate the ball with high accuracy. However, the robots do not have the ability to detect other robots. Building on the work of [Liu et al.](https://arxiv.org/abs/1512.02325) I set out to train a Single Shot Detection network that detects both balls and robots. Following the approach by the current world champion I scale the 

# Localization in the RoboCup SPL

# Reinforcement learning of robotic motion

# Symbolic regression -->
