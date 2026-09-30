/**
 * "Recommended for you": fetches this visitor's personalized products from
 * the module's JSON endpoint and renders them into the container.
 *
 * Fetched client-side because the page is full-page-cached and shared by
 * everyone, while the products are personal. Renders nothing (the container
 * stays empty) if there are no recommendations or the request fails.
 *
 * Built with DOM APIs / textContent, never innerHTML, so product names can't
 * inject markup.
 */
define(['jquery'], function ($) {
    'use strict';

    function el(tag, className, text) {
        var node = document.createElement(tag);

        if (className) {
            node.className = className;
        }
        if (text !== undefined) {
            node.textContent = text;
        }

        return node;
    }

    function renderItem(item) {
        var li = el('li', 'visualsearch-recs__item'),
            link = el('a', 'visualsearch-recs__link'),
            img = el('img');

        link.href = item.url;
        img.src = item.image;
        img.alt = item.name;
        img.loading = 'lazy';
        link.appendChild(img);
        link.appendChild(el('span', 'visualsearch-recs__name', item.name));
        li.appendChild(link);
        li.appendChild(el('span', 'visualsearch-recs__price', item.price));

        return li;
    }

    return function (config, element) {
        var params = config.productId ? {'product_id': config.productId} : {};

        $.ajax({
            url: config.url,
            type: 'GET',
            data: params,
            dataType: 'json',
            cache: false
        }).done(function (data) {
            var items = data && data.items ? data.items : [],
                section,
                list;

            if (!items.length) {
                return;
            }

            section = el('section', 'visualsearch-recs');
            section.appendChild(el('h2', 'visualsearch-recs__title', config.title));
            list = el('ul', 'visualsearch-recs__list');
            items.forEach(function (item) {
                list.appendChild(renderItem(item));
            });
            section.appendChild(list);
            element.appendChild(section);
        });
    };
});
