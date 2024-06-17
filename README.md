![ORNL Slicer 2 for next-gen additive manufacturing](doc/Slicer2.gif)

- [Design goals](#design-goals)
- [Support](#support)
  - [Report Bugs](https://github.com/ORNLSlicer/Slicer-2/issues)
  - [Releases](https://github.com/ORNLSlicer/Slicer-2/wiki/Previous%20Releases)
  - [Getting Started as a Developer](https://github.com/ORNLSlicer/Slicer-2/wiki/Getting-Started-as-a-Developer)
  - [Documentation](https://github.com/ORNLSlicer/Slicer-2/wiki/Slicer%202%20Documentation)
- [License and Copyright](#license-and-copyright)
- [Sponsors](#sponsors)
- [Contributors and Thanks](#contributors-and-thanks)
- [Engagement](#Engagement)
- [Contact](#contact)
- [Share] (#share)

## Design goals

There are numerous slicers available either open-source or commercial. In many cases, each of these slicers is attempting to solve different goals. ORNL Slicer 2.0 was designed with these goals in mind:

- **Easy Syntax Addition**. For us, as a research organization, we regularly find that a new machine does not quite fit the style of already existing machines. This can be due to a slightly different g-code flavor or the system being experimental. If a new machine is based on a g-code like output, we wanted its addition to be easy while leveraging already existing code.

- **Networking**. We look at slicing as part of a software ecosystem. The geometric evaluation of the object must interact with other pieces of software or the machines themselves. To that end, Slicer 2 is built with connectivity in mind as we push towards slicing as a platform instead of a simple program.

- **Closing the loop**.  Unlike the traditional, sequential process of slicing and preparing g-code, we push for an iterative process. The slicer should be involved in the construction of the object to ingest sensor feedback and manipulate pathing as necessary. This goes hand-in-hand with the network connectivity and creates what we call [Slice on the Fly](https://repositories.lib.utexas.edu/handle/2152/90721).
 
- **Total Control**. Slicer 2 is meant to allow complete control of your pathing process. Currently, there are nearly 500 settings to be adjusted with more constantly added. This includes not just global settings, but also settings related to each layer, each object, or specific volumes.

- **Experimental Systems**. We are a research institution. As a result, you will see code dedicated to support of experimental systems that are likely to not be found elsewhere. For example, rotary powder bed fusion, sheet lamination with pick and place, autonomous systems for construction, and novel hybrid approaches.

- **Future Goals**. Our plans are always changing. Currently, we are focused on the development of unique solutions for hybrid systems, laser powder bed, and the expansion of modularity and connectivity with frameworks such as ROS. We are also working to expand connectivity with augmented reality solutions.

## Support

If you need to report a bug :bug:, please check the [Issues page](https://github.com/ORNLSlicer/Slicer-2/issues) to see if it is known or not. If not, please feel free to submit it using the template found there.

If you need to reference documentation, both Doxygen/GraphViz class documentation and the user guide can be found [here](https://github.com/ORNLSlicer/Slicer-2/wiki/Slicer%202%20Documentation).

Help getting starting with developing can be found [here](https://github.com/ORNLSlicer/Slicer-2/wiki/Getting-Started-as-a-Developer).

## License and Copyright

License information can be found in the license file included as part of the project. For reference, that information as well as the licenses used by the various third-party libraries can be found in the [License](https://github.com/ORNLSlicer/Slicer-2/wiki/Slicer%202%20License%20and%20Library%20Licenses) section of our wiki.
Information regarding DOE [Citation and Copyright](https://github.com/ORNLSlicer/Slicer-2/wiki/Slicer%202%20Citation%20and%20Copyright%20Information) can also be found on the wiki.

## Sponsors

This work has been sponsored by work supported by the U.S. Department of Energy, Office of Energy Efficiency and Renewable Energy, Office of Advanced Manufacturing, under contract number DE-AC05-00OR22725.

## Contributors and Thanks

This work has also benefitied from cooperative research agreements with numerous partners without which Slicer 2 would not have evolved to its current state. A special mention should also be made for the many interns that have contributed to the project.

## Engagement

If expanding on this work is of interest, please feel free to contact us. We are always on the lookout for new partners! We also strongly believe in developing the pathing community. We hold a yearly seminar for all things path planning related where we bring together representatives from various partners and users to discuss topics of interest and lay out our future plans. This seminar is known as the **SL**icer **U**ser **G**roup (SLUG) and is usually held in May. Archived recordings can be found on the wiki.

## Contact

Questions regarding Slicer 2 can be directed to slicer@ornl.gov.

## Share
Need an easy way to share/connect to the repo? Use our QR code! A copy can be found in the docs folder of the repo.

![Share/Connect](doc/Slicer2QR.png)
