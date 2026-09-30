<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Block\Search;

use Magento\Framework\Data\Form\FormKey;
use Magento\Framework\View\Element\Template;
use Magento\Framework\View\Element\Template\Context;
use VisualSearch\Connector\Model\Config;

/**
 * The "search by image" upload form. Renders nothing when the module is off,
 * so the layout can add it to pages unconditionally.
 */
class Form extends Template
{
    protected $_template = 'VisualSearch_Connector::search/form.phtml';

    public function __construct(
        Context $context,
        private readonly Config $config,
        private readonly FormKey $formKey,
        array $data = []
    ) {
        parent::__construct($context, $data);
    }

    public function getActionUrl(): string
    {
        return $this->getUrl('visualsearch/search/result');
    }

    public function getFormKey(): string
    {
        return $this->formKey->getFormKey();
    }

    protected function _toHtml(): string
    {
        return $this->config->isEnabled() ? parent::_toHtml() : '';
    }
}
