/**
 * Shopping-assistant chat widget.
 *
 * Posts the shopper's message (and optional photo) to the module's JSON
 * endpoint and renders the reply plus product cards. The shopper's identity
 * is resolved server-side; the browser only keeps the conversation's
 * `session_id` (in sessionStorage) so follow-up messages have context.
 *
 * Built with DOM APIs / textContent, never innerHTML, so model replies and
 * product names can't inject markup.
 */
define(['jquery', 'mage/cookies'], function ($) {
    'use strict';

    var STORAGE_KEY = 'vs_chat_session';

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

    function readSession() {
        try {
            return window.sessionStorage.getItem(STORAGE_KEY) || '';
        } catch (e) {
            return '';
        }
    }

    function writeSession(id) {
        try {
            window.sessionStorage.setItem(STORAGE_KEY, id);
        } catch (e) {
            // Private mode etc.: the chat still works, just without memory across page loads.
        }
    }

    function productCard(item) {
        var li = el('li', 'visualsearch-chat__product'),
            link = el('a', 'visualsearch-chat__product-link'),
            img = el('img');

        link.href = item.url;
        img.src = item.image;
        img.alt = item.name;
        img.loading = 'lazy';
        link.appendChild(img);
        link.appendChild(el('span', 'visualsearch-chat__product-name', item.name));
        link.appendChild(el('span', 'visualsearch-chat__product-price', item.price));
        li.appendChild(link);

        return li;
    }

    return function (config, element) {
        var sessionId = readSession(),
            busy = false,
            toggle = el('button', 'visualsearch-chat__toggle', config.title),
            panel = el('section', 'visualsearch-chat__panel'),
            log = el('div', 'visualsearch-chat__log'),
            form = el('form', 'visualsearch-chat__form'),
            input = el('input', 'visualsearch-chat__input'),
            fileInput = el('input'),
            attach = el('button', 'visualsearch-chat__attach', '📎'),
            fileName = el('span', 'visualsearch-chat__file'),
            send = el('button', 'visualsearch-chat__send', 'Send'),
            header = el('header', 'visualsearch-chat__header', config.title),
            close = el('button', 'visualsearch-chat__close', '×');

        toggle.type = 'button';
        close.type = 'button';
        close.setAttribute('aria-label', 'Close');
        attach.type = 'button';
        attach.title = 'Attach a photo';
        send.type = 'submit';
        input.type = 'text';
        input.maxLength = 2000;
        input.placeholder = config.placeholder;
        input.setAttribute('aria-label', config.placeholder);
        fileInput.type = 'file';
        fileInput.accept = 'image/jpeg,image/png,image/webp';
        fileInput.hidden = true;
        panel.hidden = true;
        log.setAttribute('aria-live', 'polite');

        header.appendChild(close);
        form.appendChild(attach);
        form.appendChild(input);
        form.appendChild(send);
        form.appendChild(fileInput);
        panel.appendChild(header);
        panel.appendChild(log);
        panel.appendChild(fileName);
        panel.appendChild(form);
        element.appendChild(toggle);
        element.appendChild(panel);

        function addMessage(role, text) {
            var node = el('div', 'visualsearch-chat__msg visualsearch-chat__msg--' + role, text);

            log.appendChild(node);
            log.scrollTop = log.scrollHeight;

            return node;
        }

        function addProducts(products) {
            var list;

            if (!products || !products.length) {
                return;
            }
            list = el('ul', 'visualsearch-chat__products');
            products.forEach(function (item) {
                list.appendChild(productCard(item));
            });
            log.appendChild(list);
            log.scrollTop = log.scrollHeight;
        }

        function setBusy(value) {
            busy = value;
            send.disabled = value;
            input.disabled = value;
            attach.disabled = value;
        }

        function clearFile() {
            fileInput.value = '';
            fileName.textContent = '';
        }

        toggle.addEventListener('click', function () {
            panel.hidden = !panel.hidden;
            if (!panel.hidden) {
                if (!log.childNodes.length) {
                    addMessage('assistant', config.greeting);
                }
                input.focus();
            }
        });
        close.addEventListener('click', function () {
            panel.hidden = true;
        });
        attach.addEventListener('click', function () {
            fileInput.click();
        });
        fileInput.addEventListener('change', function () {
            fileName.textContent = fileInput.files.length ? fileInput.files[0].name : '';
        });

        form.addEventListener('submit', function (event) {
            var message = input.value.trim(),
                data,
                pending;

            event.preventDefault();
            if (busy || !message) {
                return;
            }

            addMessage('user', message + (fileInput.files.length ? ' [' + fileInput.files[0].name + ']' : ''));
            data = new FormData();
            data.append('message', message);
            data.append('form_key', $.mage.cookies.get('form_key') || '');
            if (sessionId) {
                data.append('session_id', sessionId);
            }
            if (fileInput.files.length) {
                data.append('image', fileInput.files[0]);
            }
            input.value = '';
            clearFile();
            setBusy(true);
            pending = addMessage('assistant', '…');

            $.ajax({
                url: config.url,
                type: 'POST',
                data: data,
                processData: false,
                contentType: false,
                dataType: 'json',
                cache: false
            }).done(function (response) {
                if (response.session_id) {
                    sessionId = response.session_id;
                    writeSession(sessionId);
                }
                pending.textContent = response.reply || '';
                addProducts(response.products);
            }).fail(function (xhr) {
                var error = xhr.responseJSON && xhr.responseJSON.error;

                pending.textContent = error || 'Sorry, something went wrong. Please try again.';
                pending.classList.add('visualsearch-chat__msg--error');
            }).always(function () {
                setBusy(false);
                input.focus();
            });
        });
    };
});
