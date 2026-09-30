/**
 * Product-page view beacon: tells the module's controller "this visitor viewed
 * product N". Sent from the browser because product pages are usually served
 * from the full-page cache and never reach PHP (see Controller/Event/Track.php).
 *
 * Counted once per product per browser session, so refreshing the page
 * doesn't inflate the signal. Failures are ignored: tracking must never
 * affect the page.
 */
define(['jquery', 'mage/cookies'], function ($) {
    'use strict';

    return function (config) {
        var key = 'vs_viewed_' + config.productId;

        try {
            if (window.sessionStorage.getItem(key)) {
                return;
            }
            window.sessionStorage.setItem(key, '1');
        } catch (e) {
            // sessionStorage unavailable (private mode / blocked): just send it.
        }

        $.post(config.url, {
            'product_id': config.productId,
            'form_key': $.mage.cookies.get('form_key')
        });
    };
});
