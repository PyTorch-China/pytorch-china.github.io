---
layout: get_started
title: 使用云服务
permalink: /get-started/cloud-partners/
background-class: get-started-background
body-class: get-started
order: 3
published: true
get-started-via-cloud: true
---

## 使用云服务

<div class="container-fluid quick-start-module quick-starts">
  <div class="cloud-options-col">
    <p>
    云平台提供强大的硬件和基础设施，用于训练和部署深度学习模型。选择下面的云平台开始使用 PyTorch。
    </p>
    {% include quick_start_cloud_options.html %}
  </div>
</div>

---

{% capture aws %}
{% include_relative installation/aws.md %}
{% endcapture %}

{% capture azure %}
{% include_relative installation/azure.md %}
{% endcapture %}

{% capture google-cloud %}
{% include_relative installation/google-cloud.md %}
{% endcapture %}

<div id="cloud">
  <div class="platform aws">{{aws | markdownify }}</div>
  <div class="platform google-cloud">{{google-cloud | markdownify }}</div>
  <div class="platform microsoft-azure">{{azure | markdownify }}</div>
</div>

<script page-id="get-started-via-cloud-partners" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>
<script src="{{ site.baseurl }}/assets/get-started-sidebar.js"></script>
