##########################################################################################
# polymath/extensions/attr_ops.py: Custom attribute operations
##########################################################################################

__all__ = ['add_attr']

# Attributes beginning with this prefix are reserved for derivatives, which are exposed as
# this prefix plus the derivative's key.
_DERIV_PREFIX = 'd_d'


def add_attr(self, name, value=None):
    """Add a custom attribute to this object and assign its value.

    The attribute becomes readable and writable as `object.name`, and it is carried along
    by every copy and clone of this object. The value is transferred by reference, not
    copied. An operation that computes new values, such as a negation or a
    multiplication, returns an object that does not carry the attribute.

    An attribute that this object already has for any other reason cannot be replaced;
    only an attribute previously added by this method can be given a new value. Names
    beginning with "d_d" are reserved for derivatives and are never allowed.

    Parameters:
        name (str): The name of the attribute, which must be a valid Python identifier
            and must not begin with "d_d".
        value (object, optional): The value of the attribute; None by default.

    Returns:
        Qube: This object after the attribute has been added.

    Raises:
        TypeError: If `name` is not a string.
        ValueError: If `name` is not a valid Python identifier, if it begins with "d_d",
            or if this object already has an attribute of this name that was not added by
            this method.
    """

    if not isinstance(name, str):
        raise TypeError(f'attribute name is not a string: {name!r}')

    if not name.isidentifier():
        raise ValueError(f'invalid attribute name: "{name}"')

    if name.startswith(_DERIV_PREFIX):
        raise ValueError(f'attribute name "{name}" is reserved for derivatives')

    added = self._added_attrs
    if name not in added:
        # The class is queried instead of the object so that no property is evaluated
        if hasattr(type(self), name) or name in self.__dict__:
            raise ValueError(f'attribute "{name}" already exists in '
                             f'{type(self).__name__} object')

        # A frozenset is never modified in place, so copies of this object can share it
        self._added_attrs = added | {name}

    setattr(self, name, value)

    # Cached objects such as "wod" were derived before this attribute existed
    self._cache.clear()
    return self

##########################################################################################
