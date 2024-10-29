#pragma once

#include <Dataset.h>
#include <PluginFactory.h>
#include <TransformationPlugin.h>

#include <actions/DatasetPickerAction.h>
#include <actions/GroupAction.h>
#include <actions/TriggerAction.h>

#include <QDialog>

class SelectInputDataDialog : public QDialog
{
    Q_OBJECT

public:
    SelectInputDataDialog(QWidget* parentWidget, const mv::Datasets& parents);

    /** Get preferred size */
    QSize sizeHint() const override {
        return QSize(400, 50);
    }

    /** Get minimum size hint*/
    QSize minimumSizeHint() const override {
        return sizeHint();
    }

    mv::Dataset<mv::DatasetImpl> getParentData() {
        return _parentsAction.getCurrentDataset();
    }

private:
    mv::gui::DatasetPickerAction    _parentsAction;
    mv::gui::TriggerAction          _loadAction;
    mv::gui::GroupAction            _groupAction;
};


/**
 * Cluster Means plugin
 *
 * Creates a new dataset with the means of each cluster
 *
 * @authors Alex Vieth
 */
class ClusterMeansPlugin : public mv::plugin::TransformationPlugin
{
    Q_OBJECT

public:

    ClusterMeansPlugin(const mv::plugin::PluginFactory* factory);
    ~ClusterMeansPlugin() override = default;

    void init() override {};

    void transform() override;
};

// =============================================================================
// Plugin Factory 
// =============================================================================

class ClusterMeansPluginFactory : public mv::plugin::TransformationPluginFactory
{
    Q_INTERFACES(mv::plugin::TransformationPluginFactory mv::plugin::PluginFactory)
        Q_OBJECT
        Q_PLUGIN_METADATA(IID   "studio.manivault.ClusterMeansPlugin"
            FILE  "ClusterMeansPlugin.json")

public:

    /** Default constructor */
    ClusterMeansPluginFactory() {}

    /** Destructor */
    ~ClusterMeansPluginFactory() override {}

    /** Creates an instance of the example analysis plugin */
    mv::plugin::TransformationPlugin* produce() override;

    /** Returns the data types that are supported by the example analysis plugin */
    mv::DataTypes supportedDataTypes() const override;

    /** Enable right-click on data set to open analysis */
    mv::gui::PluginTriggerActions getPluginTriggerActions(const mv::Datasets& datasets) const override;
};