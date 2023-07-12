<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis simplifyMaxScale="1" simplifyDrawingTol="1" maxScale="0" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" labelsEnabled="0" simplifyAlgorithm="0" simplifyLocal="1" version="3.10.4-A Coruña" simplifyDrawingHints="0" readOnly="0" minScale="1e+08">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <renderer-v2 enableorderby="0" type="singleSymbol" forceraster="0" symbollevels="0">
    <symbols>
      <symbol alpha="1" force_rhr="0" type="marker" name="0" clip_to_extent="1">
        <layer locked="0" enabled="1" class="VectorField" pass="0">
          <prop v="0" k="angle_orientation"/>
          <prop v="0" k="angle_units"/>
          <prop v="3x:0,0,0,0,0,0" k="distance_map_unit_scale"/>
          <prop v="MM" k="distance_unit"/>
          <prop v="0,0" k="offset"/>
          <prop v="3x:0,0,0,0,0,0" k="offset_map_unit_scale"/>
          <prop v="MM" k="offset_unit"/>
          <prop v="1" k="scale"/>
          <prop v="2" k="size"/>
          <prop v="3x:0,0,0,0,0,0" k="size_map_unit_scale"/>
          <prop v="MM" k="size_unit"/>
          <prop v="0" k="vector_field_type"/>
          <prop v="dx_in_meters" k="x_attribute"/>
          <prop v="dy_in_meters" k="y_attribute"/>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" name="name" value=""/>
              <Option name="properties"/>
              <Option type="QString" name="type" value="collection"/>
            </Option>
          </data_defined_properties>
          <symbol alpha="1" force_rhr="0" type="line" name="@0@0" clip_to_extent="1">
            <layer locked="0" enabled="1" class="ArrowLine" pass="0">
              <prop v="1" k="arrow_start_width"/>
              <prop v="MM" k="arrow_start_width_unit"/>
              <prop v="3x:0,0,0,0,0,0" k="arrow_start_width_unit_scale"/>
              <prop v="0" k="arrow_type"/>
              <prop v="1" k="arrow_width"/>
              <prop v="MM" k="arrow_width_unit"/>
              <prop v="3x:0,0,0,0,0,0" k="arrow_width_unit_scale"/>
              <prop v="1.5" k="head_length"/>
              <prop v="MM" k="head_length_unit"/>
              <prop v="3x:0,0,0,0,0,0" k="head_length_unit_scale"/>
              <prop v="1.5" k="head_thickness"/>
              <prop v="MM" k="head_thickness_unit"/>
              <prop v="3x:0,0,0,0,0,0" k="head_thickness_unit_scale"/>
              <prop v="0" k="head_type"/>
              <prop v="1" k="is_curved"/>
              <prop v="1" k="is_repeated"/>
              <prop v="0" k="offset"/>
              <prop v="MM" k="offset_unit"/>
              <prop v="3x:0,0,0,0,0,0" k="offset_unit_scale"/>
              <prop v="0" k="ring_filter"/>
              <data_defined_properties>
                <Option type="Map">
                  <Option type="QString" name="name" value=""/>
                  <Option name="properties"/>
                  <Option type="QString" name="type" value="collection"/>
                </Option>
              </data_defined_properties>
              <symbol alpha="1" force_rhr="0" type="fill" name="@@0@0@0" clip_to_extent="1">
                <layer locked="0" enabled="1" class="SimpleFill" pass="0">
                  <prop v="3x:0,0,0,0,0,0" k="border_width_map_unit_scale"/>
                  <prop v="0,0,255,255" k="color"/>
                  <prop v="bevel" k="joinstyle"/>
                  <prop v="0,0" k="offset"/>
                  <prop v="3x:0,0,0,0,0,0" k="offset_map_unit_scale"/>
                  <prop v="MM" k="offset_unit"/>
                  <prop v="35,35,35,255" k="outline_color"/>
                  <prop v="solid" k="outline_style"/>
                  <prop v="0.26" k="outline_width"/>
                  <prop v="MM" k="outline_width_unit"/>
                  <prop v="solid" k="style"/>
                  <data_defined_properties>
                    <Option type="Map">
                      <Option type="QString" name="name" value=""/>
                      <Option name="properties"/>
                      <Option type="QString" name="type" value="collection"/>
                    </Option>
                  </data_defined_properties>
                </layer>
              </symbol>
            </layer>
          </symbol>
        </layer>
      </symbol>
    </symbols>
    <rotation/>
    <sizescale/>
  </renderer-v2>
  <customproperties>
    <property key="dualview/previewExpressions">
      <value>fid</value>
    </property>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="variableNames"/>
    <property key="variableValues"/>
  </customproperties>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerOpacity>1</layerOpacity>
  <SingleCategoryDiagramRenderer diagramType="Histogram" attributeLegend="1">
    <DiagramCategory lineSizeType="MM" sizeScale="3x:0,0,0,0,0,0" backgroundColor="#ffffff" rotationOffset="270" opacity="1" labelPlacementMethod="XHeight" penColor="#000000" penAlpha="255" scaleBasedVisibility="0" penWidth="0" height="15" barWidth="5" minScaleDenominator="0" backgroundAlpha="255" enabled="0" scaleDependency="Area" minimumSize="0" maxScaleDenominator="1e+08" sizeType="MM" lineSizeScale="3x:0,0,0,0,0,0" width="15" diagramOrientation="Up">
      <fontProperties description="Ubuntu,11,-1,5,50,0,0,0,0,0" style=""/>
      <attribute color="#000000" label="" field=""/>
    </DiagramCategory>
  </SingleCategoryDiagramRenderer>
  <DiagramLayerSettings priority="0" obstacle="0" showAll="1" zIndex="0" dist="0" placement="0" linePlacementFlags="18">
    <properties>
      <Option type="Map">
        <Option type="QString" name="name" value=""/>
        <Option name="properties"/>
        <Option type="QString" name="type" value="collection"/>
      </Option>
    </properties>
  </DiagramLayerSettings>
  <geometryOptions removeDuplicateNodes="0" geometryPrecision="0">
    <activeChecks/>
    <checkConfiguration/>
  </geometryOptions>
  <fieldConfiguration>
    <field name="fid">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="dx_in_pixels">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="dy_in_pixels">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="dx_in_meters">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="dy_in_meters">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="d_in_meters">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="dx_in_meters_per_year">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="dy_in_meters_per_year">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="d_in_meters_per_year">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="direction">
      <editWidget type="TextEdit">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="" field="fid"/>
    <alias index="1" name="" field="dx_in_pixels"/>
    <alias index="2" name="" field="dy_in_pixels"/>
    <alias index="3" name="" field="dx_in_meters"/>
    <alias index="4" name="" field="dy_in_meters"/>
    <alias index="5" name="" field="d_in_meters"/>
    <alias index="6" name="" field="dx_in_meters_per_year"/>
    <alias index="7" name="" field="dy_in_meters_per_year"/>
    <alias index="8" name="" field="d_in_meters_per_year"/>
    <alias index="9" name="" field="direction"/>
  </aliases>
  <excludeAttributesWMS/>
  <excludeAttributesWFS/>
  <defaults>
    <default applyOnUpdate="0" field="fid" expression=""/>
    <default applyOnUpdate="0" field="dx_in_pixels" expression=""/>
    <default applyOnUpdate="0" field="dy_in_pixels" expression=""/>
    <default applyOnUpdate="0" field="dx_in_meters" expression=""/>
    <default applyOnUpdate="0" field="dy_in_meters" expression=""/>
    <default applyOnUpdate="0" field="d_in_meters" expression=""/>
    <default applyOnUpdate="0" field="dx_in_meters_per_year" expression=""/>
    <default applyOnUpdate="0" field="dy_in_meters_per_year" expression=""/>
    <default applyOnUpdate="0" field="d_in_meters_per_year" expression=""/>
    <default applyOnUpdate="0" field="direction" expression=""/>
  </defaults>
  <constraints>
    <constraint unique_strength="1" exp_strength="0" constraints="3" notnull_strength="1" field="fid"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="dx_in_pixels"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="dy_in_pixels"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="dx_in_meters"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="dy_in_meters"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="d_in_meters"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="dx_in_meters_per_year"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="dy_in_meters_per_year"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="d_in_meters_per_year"/>
    <constraint unique_strength="0" exp_strength="0" constraints="0" notnull_strength="0" field="direction"/>
  </constraints>
  <constraintExpressions>
    <constraint desc="" exp="" field="fid"/>
    <constraint desc="" exp="" field="dx_in_pixels"/>
    <constraint desc="" exp="" field="dy_in_pixels"/>
    <constraint desc="" exp="" field="dx_in_meters"/>
    <constraint desc="" exp="" field="dy_in_meters"/>
    <constraint desc="" exp="" field="d_in_meters"/>
    <constraint desc="" exp="" field="dx_in_meters_per_year"/>
    <constraint desc="" exp="" field="dy_in_meters_per_year"/>
    <constraint desc="" exp="" field="d_in_meters_per_year"/>
    <constraint desc="" exp="" field="direction"/>
  </constraintExpressions>
  <expressionfields/>
  <attributeactions>
    <defaultAction key="Canvas" value="{00000000-0000-0000-0000-000000000000}"/>
  </attributeactions>
  <attributetableconfig actionWidgetStyle="dropDown" sortExpression="&quot;dx_in_pixels&quot;" sortOrder="0">
    <columns>
      <column width="-1" type="field" name="fid" hidden="0"/>
      <column width="-1" type="actions" hidden="1"/>
      <column width="-1" type="field" name="dx_in_pixels" hidden="0"/>
      <column width="-1" type="field" name="dy_in_pixels" hidden="0"/>
      <column width="-1" type="field" name="dx_in_meters" hidden="0"/>
      <column width="-1" type="field" name="dy_in_meters" hidden="0"/>
      <column width="-1" type="field" name="d_in_meters" hidden="0"/>
      <column width="-1" type="field" name="dx_in_meters_per_year" hidden="0"/>
      <column width="-1" type="field" name="dy_in_meters_per_year" hidden="0"/>
      <column width="-1" type="field" name="d_in_meters_per_year" hidden="0"/>
      <column width="-1" type="field" name="direction" hidden="0"/>
    </columns>
  </attributetableconfig>
  <conditionalstyles>
    <rowstyles/>
    <fieldstyles/>
  </conditionalstyles>
  <storedexpressions/>
  <editform tolerant="1"></editform>
  <editforminit/>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[# -*- coding: utf-8 -*-
"""
QGIS forms can have a Python function that is called when the form is
opened.

Use this function to add extra logic to your forms.

Enter the name of the function in the "Python Init function"
field.
An example follows:
"""
from qgis.PyQt.QtWidgets import QWidget

def my_form_open(dialog, layer, feature):
	geom = feature.geometry()
	control = dialog.findChild(QWidget, "MyLineEdit")
]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>generatedlayout</editorlayout>
  <editable>
    <field name="b0" editable="1"/>
    <field name="b1" editable="1"/>
    <field name="b2" editable="1"/>
    <field name="b3" editable="1"/>
    <field name="b4" editable="1"/>
    <field name="b5" editable="1"/>
    <field name="b6" editable="1"/>
    <field name="b7" editable="1"/>
    <field name="b8" editable="1"/>
    <field name="d_in_meters" editable="1"/>
    <field name="d_in_meters_per_year" editable="1"/>
    <field name="direction" editable="1"/>
    <field name="dx_in_meters" editable="1"/>
    <field name="dx_in_meters_per_year" editable="1"/>
    <field name="dx_in_pixels" editable="1"/>
    <field name="dy_in_meters" editable="1"/>
    <field name="dy_in_meters_per_year" editable="1"/>
    <field name="dy_in_pixels" editable="1"/>
    <field name="fid" editable="1"/>
  </editable>
  <labelOnTop>
    <field labelOnTop="0" name="b0"/>
    <field labelOnTop="0" name="b1"/>
    <field labelOnTop="0" name="b2"/>
    <field labelOnTop="0" name="b3"/>
    <field labelOnTop="0" name="b4"/>
    <field labelOnTop="0" name="b5"/>
    <field labelOnTop="0" name="b6"/>
    <field labelOnTop="0" name="b7"/>
    <field labelOnTop="0" name="b8"/>
    <field labelOnTop="0" name="d_in_meters"/>
    <field labelOnTop="0" name="d_in_meters_per_year"/>
    <field labelOnTop="0" name="direction"/>
    <field labelOnTop="0" name="dx_in_meters"/>
    <field labelOnTop="0" name="dx_in_meters_per_year"/>
    <field labelOnTop="0" name="dx_in_pixels"/>
    <field labelOnTop="0" name="dy_in_meters"/>
    <field labelOnTop="0" name="dy_in_meters_per_year"/>
    <field labelOnTop="0" name="dy_in_pixels"/>
    <field labelOnTop="0" name="fid"/>
  </labelOnTop>
  <widgets/>
  <previewExpression>fid</previewExpression>
  <mapTip></mapTip>
  <layerGeometryType>0</layerGeometryType>
</qgis>
