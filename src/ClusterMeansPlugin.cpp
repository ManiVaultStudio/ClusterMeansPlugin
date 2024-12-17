#include "ClusterMeansPlugin.h"

#include <ClusterData/Cluster.h>
#include <ClusterData/ClusterData.h>
#include <PointData/PointData.h>
#include <CoreInterface.h>
#include <Dataset.h>
#include <DataType.h>
#include <LinkedData.h>

#include <ankerl/unordered_dense.h>

#include <cassert>
#include <cmath>        // sqrt
#include <cstdint>
#include <utility>      // move
#include <vector>

#include <QList>
#include <QString>
#include <QVector>

using DenseSet = ankerl::unordered_dense::set<std::int32_t>;

Q_PLUGIN_METADATA(IID "studio.manivault.ClusterMeansPlugin")

SelectInputDataDialog::SelectInputDataDialog(QWidget* parentWidget, const mv::Datasets& parents) :
    QDialog(parentWidget),
    _parentsAction(this, "Dataset"),
    _loadAction(this, "Create Dataset"),
    _groupAction(this, "Settings")
{
    setWindowTitle(tr("Compute means from..."));

    _parentsAction.setDatasets(parents);

    _groupAction.addAction(&_parentsAction);
    _groupAction.addAction(&_loadAction);

    auto layout = new QVBoxLayout();
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(_groupAction.createWidget(this));
    setLayout(layout);

    connect(&_loadAction, &mv::gui::TriggerAction::triggered, this, [this]() {
        accept();
        });
}

ClusterMeansPlugin::ClusterMeansPlugin(const PluginFactory* factory) :
    TransformationPlugin(factory)
{
}

void ClusterMeansPlugin::transform()
{
    // Get input
    auto clusterData = getInputDataset<Clusters>();

    // check if data set covers base data completely and fully
    DenseSet unique_elements;
    size_t maxDataID = 0;

    auto getGetUniqueAndDuplicates = [&maxDataID](const mv::Dataset<Clusters>& clusterData, DenseSet& unique_elements) -> bool
        {
            for (const auto& cluster : clusterData->getClusters()) {
                for (const auto id : cluster.getIndices())
                {
                    if(id > maxDataID)
                        maxDataID = id;

                    // Insert failed, indicating a duplicate
                    if (!unique_elements.insert(id).second)
                        return true;
                }
            }
            return false;
        };

    if (getGetUniqueAndDuplicates(clusterData, unique_elements))
    {
        qDebug() << "Contains duplicate IDs, cannot create means data";
        return;
    }

    size_t numClusterIDs = unique_elements.size();

    mv::Datasets possibleParents;
    for (auto parentItem : clusterData->getDataHierarchyItem().getAncestors())
    {
        if (parentItem == nullptr)
            continue;

        auto parentDataset = parentItem->getDataset();

        // we want points and full data
        if (parentDataset.getDataset()->getDataType() != PointType || !parentDataset.getDataset()->isFull())
            continue;

        const auto parentPoints = mv::Dataset<Points>(parentDataset);
        if (!parentPoints.isValid())
            continue;

        auto numDataPoints = parentPoints->getNumPoints();
        if (numDataPoints != numClusterIDs && numDataPoints < maxDataID)
            continue;

        possibleParents << parentDataset;
    }

    // datasets with the same selection group index as one of the parents might also be interesting
    DenseSet selectionGroupIDs;
    for (const auto& possibleParent : possibleParents)
        if (possibleParent->getGroupIndex() >= 0)
            selectionGroupIDs.insert(possibleParent->getGroupIndex());

    if (selectionGroupIDs.size() > 0) {
        for (const auto& selectionGroupID : selectionGroupIDs) {
            for (const auto& dataset : mv::data().getAllDatasets()) {
                if (dataset->getGroupIndex() == selectionGroupID && dataset.getDataset()->getDataType() == PointType) {
                    const auto datasetPoints = mv::Dataset<Points>(dataset);
                    if (datasetPoints.isValid() && datasetPoints->getNumPoints() != numClusterIDs)
                        continue;

                    if (possibleParents.contains(dataset))
                        continue;

                    possibleParents << dataset;
                }
            }
        }
    }

    // Ask user for dataset
    SelectInputDataDialog inputDialog(nullptr, possibleParents);
    inputDialog.setModal(true);

    // open dialog and wait for user input
    int ok = inputDialog.exec();

    if (ok == QDialog::Accepted ) {

        auto parentDataset = inputDialog.getParentData();

        if (!parentDataset.isValid())
        {
            qDebug() << "ClusterMeans: Please select a valid data set";
            return;
        }

        auto parentPoints = mv::Dataset<Points>(parentDataset);
        QVector<Cluster>& clusters = clusterData->getClusters();

        // Compute means and standard deviations
        auto numDims = parentPoints->getNumDimensions();
        auto numPoints = parentPoints->getNumPoints();
        auto numClusters = clusters.size();

        qDebug() << "ClusterMeans: using " << parentDataset->getGuiName() << " with " << numDims << " dimension";

        for (Cluster& cluster : clusters)
        {
            const auto& indices = cluster.getIndices();
            std::vector<float>& average = cluster.getMean();
            std::vector<float>& stddev = cluster.getStandardDeviation();

            average.resize(numDims);
            stddev.resize(numDims);

            parentPoints->visitData([this, numPoints, numDims, &indices, &average, &stddev](auto pointData) {
                // averages
                for (const auto& i: indices) {
                    const auto& values = pointData[i];
                    for (int64_t dim = 0; dim < numDims; ++dim) {
                        average[dim] += values[dim];
                    }
                }

                for (auto& averageDim : average) {
                    averageDim /= static_cast<float>(indices.size());
                }

                // standard deviation
                for (const auto& i : indices) {
                    const auto& values = pointData[i];

                    for (int64_t dim = 0; dim < numDims; ++dim) {
                        const auto centered = values[dim] - average[dim];
                        stddev[dim] += (centered * centered);
                    }
                }

                for (auto& stddevDim : stddev) {
                    stddevDim = std::sqrt(stddevDim / static_cast<float>(indices.size()));
                }

                });

        }

        // Create new output
        Dataset<Points> meansData = mv::data().createDataset<Points>("Points", parentDataset->getGuiName() + " Means", parentDataset);

        // Get means from clusters
        std::vector<float> means;
        for (const auto& cluster : clusters)
        {
            const auto& avgs = cluster.getMean();
            means.insert(means.end(), avgs.begin(), avgs.end());
        }

        assert(means.size() == numDims * numClusters);

        // Add selection maps
        mv::SelectionMap selectionMapMeansToParents;
        auto& mapMeansToParents = selectionMapMeansToParents.getMap();

        for (size_t clusterID = 0; clusterID < numClusters; clusterID++)
            mapMeansToParents[clusterID] = clusters[clusterID].getIndices();

        meansData->addLinkedData(parentDataset, selectionMapMeansToParents);

        // Publish data
        meansData->setData(std::move(means), numDims);
        events().notifyDatasetDataChanged(meansData);

    }

}


// =============================================================================
// Plugin Factory 
// =============================================================================

TransformationPlugin* ClusterMeansPluginFactory::produce()
{
    return new ClusterMeansPlugin(this);
}

mv::DataTypes ClusterMeansPluginFactory::supportedDataTypes() const
{
    return { ClusterType };
}

mv::gui::PluginTriggerActions ClusterMeansPluginFactory::getPluginTriggerActions(const mv::Datasets& datasets) const
{
    mv::gui::PluginTriggerActions pluginTriggerActions;

    const auto numberOfDatasets = datasets.count();

    if (PluginFactory::areAllDatasetsOfTheSameType(datasets, ClusterType)) {
        if (numberOfDatasets >= 1 && datasets.first()->getDataType() == ClusterType) {

            auto pluginTriggerAction = new mv::gui::PluginTriggerAction(const_cast<ClusterMeansPluginFactory*>(this), this, QString("Create Mean Dataset"), QString("Create Mean Dataset"), getIcon(), [this, datasets](mv::gui::PluginTriggerAction& pluginTriggerAction) -> void {
                for (const auto& dataset : datasets) {
                    auto pluginInstance = dynamic_cast<ClusterMeansPlugin*>(plugins().requestPlugin(getKind()));

                    pluginInstance->setInputDataset(dataset);
                    pluginInstance->transform();
                }
                });

            pluginTriggerActions << pluginTriggerAction;

        }
    }

    return pluginTriggerActions;
}