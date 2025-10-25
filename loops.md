# Magnetic field of circular loop

According to [Hampton et
al](https://pubs.aip.org/aip/adv/article/10/6/065320/997382/Closed-form-expressions-for-the-magnetic-fields-of),
a circular loop in $(x,y)$ plane centered at the origin with radius
*a*, the field in cylindrical coordinates $(r,\theta,z)$ is

$$B_r=\frac{az}{\pi r\sqrt{(a+r)^2+z^2}}\left(\frac{a^2+r^2+z^2}{(a-r)^2+z^2}E(m)-K(m)\right),$$
$$B_r=\frac{a}{\pi\sqrt{(a+r)^2+z^2}}\left(\frac{a^2+r^2+z^2}{(a-r)^2+z^2}E(m)+K(m)\right),$$
$$B_{\theta}=0,$$
$$m=\frac{4ar}{(a+r)^2+z^2},$$

where $K(m)$ and $E(m)$ are complete elliptic integrals of [the
first](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html) and [the
second](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html#scipy.special.ellipe)
kind.
